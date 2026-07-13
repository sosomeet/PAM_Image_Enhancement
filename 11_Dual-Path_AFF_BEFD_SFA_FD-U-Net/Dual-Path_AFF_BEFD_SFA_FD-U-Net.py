from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

SEED = 42

# ------------------------------------------------------------
# Data paths
# ------------------------------------------------------------
#
# IMPORTANT:
# This training script does NOT create 3-adjacent Y-Z data.
# It only loads the already-generated LOW files:
#
#   LOW  : [X, C, Y, Z] = [200, 3, 200, 512]
#   HIGH : [X, Y, Z]    = [200, 200, 512]
#
# LOW files must therefore already exist in:
#   ./data/3d_Train/LOW
#
# ------------------------------------------------------------

DATA_ROOT = Path("./data")
TRAIN_LOW_DIR = DATA_ROOT / "3d_Train" / "LOW"
TRAIN_HIGH_DIR = DATA_ROOT / "norm_Train" / "HIGH"

# ------------------------------------------------------------
# Binary data specification
# ------------------------------------------------------------

HEADER_SIZE = 48
DATA_DTYPE = np.float32

# HIGH normalized volume:
# [X, Y, Z]
HIGH_VOLUME_SHAPE = (200, 200, 512)

# Pre-generated 3-adjacent Y-Z LOW volume:
# [X, C, Y, Z]
ADJACENT_LOW_SHAPE = (200, 3, 200, 512)

X_SIZE = 200
Y_SIZE = 200
Z_SIZE = 512

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

INPUT_SAMPLE_SHAPE = (3, Y_SIZE, Z_SIZE)
TARGET_SAMPLE_SHAPE = (1, Y_SIZE, Z_SIZE)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_EPOCHS = 50
LEARNING_RATE = 2e-4

# Each epoch randomly samples this many X positions per volume.
# Example:
# 130 volumes x 32 X positions = 4,160 samples / epoch.
SLICES_PER_VOLUME_PER_EPOCH = 32

BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0

USE_AMP = True
USE_CHANNELS_LAST = True

# Update tqdm postfix only every N steps.
LOG_EVERY = 20

# ------------------------------------------------------------
# U-Net
# ------------------------------------------------------------

BASE_CHANNELS = 8
MAX_CHANNELS = 128

# ------------------------------------------------------------
# E2 backbone retained: Scale-aware Feature Aggregation (SFA)
# ------------------------------------------------------------
#
# E2 keeps the E1 Fully Dense U-Net backbone and adds SFA only
# at the deeper feature stages where multi-scale context is most
# useful and the spatial maps are smaller:
#
#   Encoder 3 : 32 channels, 52 x 128
#   Encoder 4 : 64 channels, 26 x 64
#   Bottleneck: 128 channels, 13 x 32
#
# Dilation rates follow the 1, 3, 5 multi-scale setting described
# for SFA-style retinal-vessel feature aggregation.
#
SFA_DILATION_RATES = (1, 3, 5)
SFA_STAGES = ("encoder3", "encoder4", "bottleneck")

# ------------------------------------------------------------
# E3 components retained in E4: BEFD (Boundary Enhancement + Feature Denoising)
# ------------------------------------------------------------
#
# E3 keeps the complete E2 backbone and adds the two BEFD ideas:
#
#   Boundary Enhancement (BE)
#     - Fixed Sobel edge prior from the center LOW Y-Z plane.
#     - Uses the threshold/linear transform described in BEFD.
#     - Applied by element-wise multiplication to the last three
#       encoder stages: Encoder 3, Encoder 4, Bottleneck.
#
#   Feature Denoising (FD)
#     - Inserted into the first three skip connections.
#     - Uses a non-local dot-product denoising operation followed by
#       1x1 convolution and a residual connection.
#     - For RTX 3060 Ti practicality, non-local attention is computed
#       on an adaptively pooled feature grid, then resized back. This
#       preserves the BEFD non-local denoising mechanism while avoiding
#       O((H*W)^2) memory at full 200x512 resolution.
#
# Original BEFD edge parameters reported in the paper:
#   lambda_min=0.8, lambda_max=5.0, alpha=2.0, beta=1.0
#
BEFD_EDGE_LAMBDA_MIN = 0.8
BEFD_EDGE_LAMBDA_MAX = 5.0
BEFD_EDGE_ALPHA = 2.0
BEFD_EDGE_BETA = 1.0
BEFD_EDGE_EPS = 1e-6

BEFD_BOUNDARY_STAGES = (
    "encoder3",
    "encoder4",
    "bottleneck",
)

BEFD_DENOISE_SKIP_STAGES = (
    "encoder1",
    "encoder2",
    "encoder3",
)

# Maximum number of spatial positions used by each non-local
# denoising block. The pooled attention matrix is at most
# [B, 512, 512], which is practical on an RTX 3060 Ti.
BEFD_DENOISE_MAX_POSITIONS = 512

# ------------------------------------------------------------
# E4 components retained in E5: Dual-path boundary-detail branch
# ------------------------------------------------------------
#
# E5 keeps the complete E4 architecture unchanged:
#
# Main context path:
#   - 3-adjacent input [X-1, X, X+1]
#   - Fully Dense U-Net backbone
#   - SFA at Encoder 3 / Encoder 4 / Bottleneck
#   - BEFD boundary enhancement and feature denoising
#
# Boundary-detail path:
#   - Input: center LOW plane + BEFD Sobel guidance
#   - Lightweight channels: 8 -> 16 -> 32 -> 64 projection
#   - Gated boundary fusion at Decoder 4, Decoder 3, Decoder 1
#
DUAL_PATH_ENABLED = True
BOUNDARY_PATH_CHANNELS = (8, 16, 32)
BOUNDARY_FUSION_STAGES = (
    "decoder4",
    "decoder3",
    "decoder1",
)

# Negative initialization keeps the initial sigmoid gate conservative,
# reducing the chance that the detail path dominates the main context
# pathway at the beginning of training.
BOUNDARY_GATE_BIAS_INIT = -2.0

# ------------------------------------------------------------
# E5: Adaptive Feature Fusion (AFF)
# ------------------------------------------------------------
#
# E5 adds one controlled architectural variable to E4: adaptive fusion
# between the upsampled decoder feature and its corresponding encoder
# skip feature at all four decoder stages.
#
# The implementation is inspired by the SCS-Net AFF concept of
# adaptively fusing adjacent hierarchical features, but is adapted to
# this PAM image-to-image regression network.
#
# For each decoder stage:
#   1. Decoder and skip features are projected to the same C channels.
#   2. Local context and global context jointly predict two branch logits.
#   3. Softmax across the two branches yields per-channel, per-pixel weights.
#   4. Weighted decoder and skip features are concatenated, preserving the
#      original 2C input expected by the existing Dense decoder block.
#
# The final AFF logits are zero-initialized. Therefore, at initialization:
#
#   w_decoder = w_skip = 0.5
#
# and AFF_OUTPUT_SCALE=2.0 makes the initial output numerically equivalent
# in scale to the raw E4 concatenation:
#
#   concat(2 * 0.5 * decoder, 2 * 0.5 * skip)
#   = concat(decoder, skip)
#
# This makes E5 a clean controlled ablation of AFF.
#
AFF_ENABLED = True
AFF_STAGES = (
    "decoder4",
    "decoder3",
    "decoder2",
    "decoder1",
)
AFF_REDUCTION = 4
AFF_MIN_HIDDEN_CHANNELS = 4
AFF_OUTPUT_SCALE = 2.0

# Spatial height 200 is not divisible by 16.
# 200 -> pad to 208 -> 104 -> 52 -> 26 -> 13
PAD_TOP = 4
PAD_BOTTOM = 4

# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

CHECKPOINT_EVERY = 10
MAP_SAVE_EVERY = 5

# Always use the same Train volume for MAP visualization.
MAP_VOLUME_INDEX = 0
MAP_BATCH_SIZE = 16
SAVE_INITIAL_MAP = False

MODEL_ROOT = Path("./11_Dual-Path_AFF_BEFD_SFA_FD-U-Net/models_enhancement")
LATEST_MODEL_DIR = MODEL_ROOT / "latest"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
FINAL_MODEL_DIR = MODEL_ROOT / "final"

OUTPUT_ROOT = Path("./11_Dual-Path_AFF_BEFD_SFA_FD-U-Net/outputs")
TRAIN_MAP_DIR = OUTPUT_ROOT / "train_maps"
HISTORY_DIR = OUTPUT_ROOT / "history"

LATEST_MODEL_PATH = LATEST_MODEL_DIR / "model_latest.pth"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "model_final.pth"

HISTORY_CSV_PATH = HISTORY_DIR / "training_history.csv"
LOSS_CURVE_PATH = HISTORY_DIR / "training_loss_curve.png"


# ============================================================
# Reproducibility / CUDA
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return

    # Fixed input size allows cuDNN to benchmark efficient kernels.
    torch.backends.cudnn.benchmark = True

    # RTX 30-series supports TF32.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def create_output_directories() -> None:
    for directory in (
        LATEST_MODEL_DIR,
        CHECKPOINT_DIR,
        FINAL_MODEL_DIR,
        TRAIN_MAP_DIR,
        HISTORY_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# File utilities
# ============================================================


def extract_volume_key(file_path: Path) -> str:
    """
    Example:
        Train_000_LOW.bin  -> Train_000
        Train_000_HIGH.bin -> Train_000
    """
    return re.sub(
        r"_(LOW|HIGH)$",
        "",
        file_path.stem,
        flags=re.IGNORECASE,
    )


def get_bin_files(directory: Path) -> list[Path]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = sorted(directory.glob("*.bin"))

    if not files:
        raise RuntimeError(
            f"No .bin files found in: {directory}"
        )

    return files


def validate_bin_file(
    file_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype = DATA_DTYPE,
    offset: int = HEADER_SIZE,
) -> None:
    """
    Validate:
        [48-byte header] + [float32 payload]
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    dtype = np.dtype(dtype)

    expected_elements = int(np.prod(shape))
    expected_file_size = (
        offset
        + expected_elements * dtype.itemsize
    )

    actual_file_size = file_path.stat().st_size

    if actual_file_size != expected_file_size:
        raise ValueError(
            "\nBinary file size mismatch\n"
            f"File            : {file_path}\n"
            f"Actual bytes    : {actual_file_size:,}\n"
            f"Expected bytes  : {expected_file_size:,}\n"
            f"Expected shape  : {shape}\n"
            f"Expected dtype  : {dtype}\n"
            f"Header size     : {offset}"
        )


def build_file_pairs(
    low_dir: Path,
    high_dir: Path,
) -> list[dict[str, Any]]:
    """
    Pair:
        Train_000_LOW.bin
        Train_000_HIGH.bin
    """

    low_files = get_bin_files(low_dir)
    high_files = get_bin_files(high_dir)

    low_map = {
        extract_volume_key(path): path
        for path in low_files
    }

    high_map = {
        extract_volume_key(path): path
        for path in high_files
    }

    low_keys = set(low_map)
    high_keys = set(high_map)

    missing_high = sorted(low_keys - high_keys)
    missing_low = sorted(high_keys - low_keys)

    if missing_high:
        raise RuntimeError(
            "LOW files without matching HIGH files:\n"
            + "\n".join(missing_high)
        )

    if missing_low:
        raise RuntimeError(
            "HIGH files without matching LOW files:\n"
            + "\n".join(missing_low)
        )

    pairs: list[dict[str, Any]] = []

    for key in sorted(low_keys & high_keys):
        low_path = low_map[key]
        high_path = high_map[key]

        # LOW is already pre-generated as [X, C, Y, Z].
        validate_bin_file(
            low_path,
            ADJACENT_LOW_SHAPE,
        )

        # HIGH remains the normalized original [X, Y, Z].
        validate_bin_file(
            high_path,
            HIGH_VOLUME_SHAPE,
        )

        pairs.append(
            {
                "volume_key": key,
                "low_path": low_path,
                "high_path": high_path,
            }
        )

    if not pairs:
        raise RuntimeError(
            "No valid LOW-HIGH pairs found."
        )

    return pairs


# ============================================================
# Dataset
# ============================================================


class PAMAdjacentYZDataset(Dataset):
    """
    Train-only PAM Dataset.

    This class does NOT create 3-adjacent planes.

    It only reads already-generated LOW files:

        LOW:
            [X, C, Y, Z]
            [200, 3, 200, 512]

        HIGH:
            [X, Y, Z]
            [200, 200, 512]

    One indexed sample at x_index:

        input:
            LOW[x_index]
            -> [3, Y, Z]
            -> [3, 200, 512]

        target:
            HIGH[x_index, :, :]
            -> [Y, Z]
            -> [1, 200, 512]

    Full dataset size:

        num_volumes x 200 X positions
    """

    def __init__(
        self,
        low_dir: Path,
        high_dir: Path,
    ) -> None:
        super().__init__()

        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)

        self.pairs = build_file_pairs(
            low_dir=self.low_dir,
            high_dir=self.high_dir,
        )

        self.num_volumes = len(self.pairs)
        self.samples_per_volume = X_SIZE
        self.total_samples = (
            self.num_volumes
            * self.samples_per_volume
        )

        # Lazy memmap caches.
        # Each DataLoader worker has its own Dataset process state.
        self._low_memmaps: dict[int, np.memmap] = {}
        self._high_memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.total_samples

    def _get_low_memmap(
        self,
        volume_index: int,
    ) -> np.memmap:
        if volume_index not in self._low_memmaps:
            low_path = self.pairs[volume_index]["low_path"]

            self._low_memmaps[volume_index] = np.memmap(
                filename=low_path,
                dtype=DATA_DTYPE,
                mode="r",
                offset=HEADER_SIZE,
                shape=ADJACENT_LOW_SHAPE,
                order="C",
            )

        return self._low_memmaps[volume_index]

    def _get_high_memmap(
        self,
        volume_index: int,
    ) -> np.memmap:
        if volume_index not in self._high_memmaps:
            high_path = self.pairs[volume_index]["high_path"]

            self._high_memmaps[volume_index] = np.memmap(
                filename=high_path,
                dtype=DATA_DTYPE,
                mode="r",
                offset=HEADER_SIZE,
                shape=HIGH_VOLUME_SHAPE,
                order="C",
            )

        return self._high_memmaps[volume_index]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Dataset index out of range: {index}. "
                f"Valid range: 0 ~ {len(self) - 1}"
            )

        volume_index, x_index = divmod(
            index,
            self.samples_per_volume,
        )

        pair = self.pairs[volume_index]

        low_volume = self._get_low_memmap(
            volume_index
        )

        high_volume = self._get_high_memmap(
            volume_index
        )

        # ----------------------------------------------------
        # Input
        #
        # Already-generated LOW:
        # [X, C, Y, Z]
        #
        # LOW[x_index]:
        # [C, Y, Z]
        # [3, 200, 512]
        #
        # No X-1/X/X+1 grouping is performed here.
        # ----------------------------------------------------

        input_array = np.array(
            low_volume[x_index, :, :, :],
            dtype=np.float32,
            copy=True,
        )

        # ----------------------------------------------------
        # Target
        #
        # HIGH:
        # [X, Y, Z]
        #
        # HIGH[x_index]:
        # [Y, Z]
        # [200, 512]
        #
        # Add channel:
        # [1, 200, 512]
        # ----------------------------------------------------

        target_array = np.array(
            high_volume[x_index, :, :],
            dtype=np.float32,
            copy=True,
        )

        target_array = np.expand_dims(
            target_array,
            axis=0,
        )

        if input_array.shape != INPUT_SAMPLE_SHAPE:
            raise ValueError(
                "\nInput shape mismatch\n"
                f"Volume key    : {pair['volume_key']}\n"
                f"X index       : {x_index}\n"
                f"Actual shape  : {input_array.shape}\n"
                f"Expected shape: {INPUT_SAMPLE_SHAPE}"
            )

        if target_array.shape != TARGET_SAMPLE_SHAPE:
            raise ValueError(
                "\nTarget shape mismatch\n"
                f"Volume key    : {pair['volume_key']}\n"
                f"X index       : {x_index}\n"
                f"Actual shape  : {target_array.shape}\n"
                f"Expected shape: {TARGET_SAMPLE_SHAPE}"
            )

        return {
            "input": torch.from_numpy(input_array),
            "target": torch.from_numpy(target_array),
            "volume_key": pair["volume_key"],
            "volume_index": volume_index,
            "x_index": x_index,
        }

    def close(self) -> None:
        self._low_memmaps.clear()
        self._high_memmaps.clear()

    def __getstate__(self) -> dict[str, Any]:
        """
        Prevent open memmaps from being serialized
        into DataLoader worker processes.
        """
        state = self.__dict__.copy()
        state["_low_memmaps"] = {}
        state["_high_memmaps"] = {}
        return state


# ============================================================
# Epoch-wise random X-position sampler
# ============================================================


class EpochRandomXSliceSampler(Sampler[int]):
    """
    For each epoch:

      1. Shuffle volume order.
      2. Sample N unique X positions per volume.
      3. Sort sampled X positions within each volume.

    Example:

        130 volumes x 32 X positions
        = 4,160 training samples / epoch

    The full Dataset still contains every X position.
    Therefore, full 200-plane inference remains possible.
    """

    def __init__(
        self,
        dataset: PAMAdjacentYZDataset,
        slices_per_volume: int,
        seed: int = 42,
    ) -> None:
        if slices_per_volume <= 0:
            raise ValueError(
                "slices_per_volume must be > 0"
            )

        if slices_per_volume > dataset.samples_per_volume:
            raise ValueError(
                f"slices_per_volume={slices_per_volume} "
                f"exceeds available X positions="
                f"{dataset.samples_per_volume}"
            )

        self.dataset = dataset
        self.slices_per_volume = slices_per_volume
        self.seed = seed
        self.epoch = 0

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (
            self.dataset.num_volumes
            * self.slices_per_volume
        )

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(
            self.seed + self.epoch
        )

        volume_order = rng.permutation(
            self.dataset.num_volumes
        )

        global_indices: list[int] = []

        for volume_index in volume_order:
            x_indices = rng.choice(
                self.dataset.samples_per_volume,
                size=self.slices_per_volume,
                replace=False,
            )

            # Improve disk/page-cache locality.
            x_indices.sort()

            base_index = (
                int(volume_index)
                * self.dataset.samples_per_volume
            )

            global_indices.extend(
                base_index + int(x_index)
                for x_index in x_indices
            )

        return iter(global_indices)


# ============================================================
# Model
# ============================================================


class DenseLayer(nn.Module):
    """
    One densely connected feature-producing layer.

    Input:
        [B, C_in, H, W]

    Output:
        [B, growth_rate, H, W]

    BatchNorm is intentionally not used so that E3 preserves the
    normalization setting of E1/E2 and isolates the added BEFD effect.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.relu(self.conv(x))


class DenseBlock(nn.Module):
    """
    Fully dense feature block inherited from E1.

    Every internal layer receives the concatenation of all previous
    features in the block:

        x1 = H1([x0])
        x2 = H2([x0, x1])
        x3 = H3([x0, x1, x2])

    A final 1x1 transition convolution compresses the concatenated
    features to the requested out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        growth_rate: int | None = None,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be > 0, got {num_layers}"
            )

        if growth_rate is None:
            growth_rate = max(4, out_channels // 2)

        if growth_rate <= 0:
            raise ValueError(
                f"growth_rate must be > 0, got {growth_rate}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate

        self.layers = nn.ModuleList()

        current_channels = in_channels

        for _ in range(num_layers):
            self.layers.append(
                DenseLayer(
                    in_channels=current_channels,
                    growth_rate=growth_rate,
                )
            )

            current_channels += growth_rate

        self.transition = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.transition_relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = [x]

        for layer in self.layers:
            dense_input = torch.cat(
                features,
                dim=1,
            )

            new_feature = layer(dense_input)
            features.append(new_feature)

        fused = torch.cat(
            features,
            dim=1,
        )

        output = self.transition(fused)
        output = self.transition_relu(output)

        return output


class AdjacentScaleSelector(nn.Module):
    """
    Dynamically fuse two adjacent receptive-field features.

    Inputs:
        feature_a: [B, C, H, W]
        feature_b: [B, C, H, W]

    A small learnable network predicts two per-channel, per-pixel
    weight maps. Softmax across the two scale branches guarantees:

        weight_a + weight_b = 1

    at every channel and spatial location.

    Output:
        [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be > 0, got {channels}"
            )

        self.channels = channels

        self.weight_predictor = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        if feature_a.shape != feature_b.shape:
            raise ValueError(
                "Adjacent scale feature shape mismatch: "
                f"{feature_a.shape} vs {feature_b.shape}"
            )

        batch_size, channels, height, width = feature_a.shape

        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}"
            )

        pair = torch.cat(
            [feature_a, feature_b],
            dim=1,
        )

        logits = self.weight_predictor(pair)

        # [B, 2C, H, W]
        # -> [B, 2, C, H, W]
        logits = logits.reshape(
            batch_size,
            2,
            channels,
            height,
            width,
        )

        weights = torch.softmax(
            logits,
            dim=1,
        )

        weight_a = weights[:, 0, :, :, :]
        weight_b = weights[:, 1, :, :, :]

        return (
            weight_a * feature_a
            + weight_b * feature_b
        )


class ScaleAwareFeatureAggregation(nn.Module):
    """
    SFA module inherited unchanged from E2 for the PAM enhancement backbone.

    The module follows the core SCS-Net SFA idea:

      1. Extract three parallel features with different dilation rates.
      2. Dynamically fuse adjacent scale pairs.
      3. Aggregate the two selected multi-scale outputs.
      4. Add a residual path from the input.

    Dilation rates:
        1, 3, 5

    Input / output:
        [B, C, H, W] -> [B, C, H, W]

    Important E3 ablation choice:
        No BatchNorm is added, so the E2 SFA behavior remains unchanged while BEFD is added separately.
    """

    def __init__(
        self,
        channels: int,
        dilation_rates: tuple[int, int, int] = SFA_DILATION_RATES,
    ) -> None:
        super().__init__()

        if len(dilation_rates) != 3:
            raise ValueError(
                "SFA requires exactly three dilation rates, "
                f"got {dilation_rates}"
            )

        if any(rate <= 0 for rate in dilation_rates):
            raise ValueError(
                f"All dilation rates must be > 0, got {dilation_rates}"
            )

        self.channels = channels
        self.dilation_rates = tuple(
            int(rate)
            for rate in dilation_rates
        )

        self.scale_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                    bias=True,
                )
                for rate in self.dilation_rates
            ]
        )

        self.scale_relu = nn.ReLU(inplace=True)

        self.select_12 = AdjacentScaleSelector(
            channels=channels,
        )

        self.select_23 = AdjacentScaleSelector(
            channels=channels,
        )

        # Final 1x1 fusion keeps the channel count unchanged.
        self.output_fusion = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.output_relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.size(1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.size(1)}"
            )

        scale_features = [
            self.scale_relu(conv(x))
            for conv in self.scale_convs
        ]

        d1, d2, d3 = scale_features

        # Adjacent receptive-field fusion:
        #   (rate 1, rate 3)
        #   (rate 3, rate 5)
        out_12 = self.select_12(d1, d2)
        out_23 = self.select_23(d2, d3)

        # Residual aggregation preserves the original dense feature
        # while adding scale-selected context.
        fused = out_12 + out_23 + x

        output = self.output_fusion(fused)
        output = self.output_relu(output)

        return output



class SobelBoundaryAttention(nn.Module):
    """
    Fixed Sobel boundary-attention map used by the E3 BEFD module.

    Input:
        center LOW plane: [B, 1, H, W]

    Sobel gradient:
        G = sqrt(Gx^2 + Gy^2)

    BEFD threshold / linear transform:

        attention = 1,
            if G > lambda_max or G < lambda_min

        attention =
            (1 - (G - lambda_min) / (lambda_max - lambda_min))
            * alpha + beta,
            otherwise

    With the paper's reported parameters:
        lambda_min = 0.8
        lambda_max = 5.0
        alpha      = 2.0
        beta       = 1.0

    Weak in-range edges therefore receive larger weights than strong
    in-range edges, while values outside the selected gradient interval
    receive a neutral multiplicative weight of 1.

    Output:
        [B, 1, H, W]
    """

    def __init__(
        self,
        lambda_min: float = BEFD_EDGE_LAMBDA_MIN,
        lambda_max: float = BEFD_EDGE_LAMBDA_MAX,
        alpha: float = BEFD_EDGE_ALPHA,
        beta: float = BEFD_EDGE_BETA,
        eps: float = BEFD_EDGE_EPS,
    ) -> None:
        super().__init__()

        if lambda_max <= lambda_min:
            raise ValueError(
                "lambda_max must be greater than lambda_min, "
                f"got {lambda_min} and {lambda_max}"
            )

        if eps <= 0:
            raise ValueError(
                f"eps must be > 0, got {eps}"
            )

        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)

        # Fixed, non-trainable kernels move automatically with the model.
        self.register_buffer(
            "sobel_x",
            sobel_x,
            persistent=True,
        )
        self.register_buffer(
            "sobel_y",
            sobel_y,
            persistent=True,
        )

    def forward(
        self,
        center: torch.Tensor,
    ) -> torch.Tensor:
        if center.ndim != 4:
            raise ValueError(
                "Expected center plane [B, 1, H, W], "
                f"got shape {tuple(center.shape)}"
            )

        if center.size(1) != 1:
            raise ValueError(
                "SobelBoundaryAttention requires one input channel, "
                f"got {center.size(1)}"
            )

        # Use float32 for a stable fixed Sobel computation under AMP.
        center_float = center.float()

        padded = F.pad(
            center_float,
            pad=(1, 1, 1, 1),
            mode="reflect",
        )

        gradient_x = F.conv2d(
            padded,
            self.sobel_x,
            padding=0,
        )

        gradient_y = F.conv2d(
            padded,
            self.sobel_y,
            padding=0,
        )

        gradient = torch.sqrt(
            gradient_x.square()
            + gradient_y.square()
            + self.eps
        )

        in_range = (
            (gradient >= self.lambda_min)
            & (gradient <= self.lambda_max)
        )

        normalized = (
            gradient - self.lambda_min
        ) / (
            self.lambda_max - self.lambda_min
        )

        transformed = (
            (1.0 - normalized) * self.alpha
            + self.beta
        )

        neutral = torch.ones_like(
            transformed
        )

        attention = torch.where(
            in_range,
            transformed,
            neutral,
        )

        return attention.to(dtype=center.dtype)


class FeatureDenoisingBlock(nn.Module):
    """
    Efficient BEFD-style non-local feature-denoising block.

    The original BEFD denoising block performs:

        1. Dot-product non-local attention over spatial positions.
        2. Softmax normalization of the pairwise affinity matrix.
        3. Non-local weighted feature aggregation.
        4. 1x1 convolution.
        5. Residual addition with the input feature.

    Direct full-resolution non-local attention is O((H*W)^2), which is
    impractical for the current PAM feature maps, especially:

        Encoder 1: 208 x 512 = 106,496 positions.

    E3 therefore computes the same dot-product non-local operation on an
    adaptively pooled grid with at most max_positions spatial locations,
    upsamples the denoised residual, and adds it back to the original
    full-resolution feature.

    Input / output:
        [B, C, H, W] -> [B, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        max_positions: int = BEFD_DENOISE_MAX_POSITIONS,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be > 0, got {channels}"
            )

        if max_positions <= 0:
            raise ValueError(
                "max_positions must be > 0, "
                f"got {max_positions}"
            )

        self.channels = int(channels)
        self.max_positions = int(max_positions)

        self.output_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

    def _get_pooled_size(
        self,
        height: int,
        width: int,
    ) -> tuple[int, int]:
        total_positions = height * width

        if total_positions <= self.max_positions:
            return height, width

        scale = (
            self.max_positions / total_positions
        ) ** 0.5

        pooled_height = max(
            1,
            int(height * scale),
        )

        pooled_width = max(
            1,
            int(width * scale),
        )

        # Numerical safety for integer rounding.
        while (
            pooled_height * pooled_width
            > self.max_positions
        ):
            if pooled_width >= pooled_height and pooled_width > 1:
                pooled_width -= 1
            elif pooled_height > 1:
                pooled_height -= 1
            else:
                break

        return pooled_height, pooled_width

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected feature [B, C, H, W], "
                f"got shape {tuple(x.shape)}"
            )

        batch_size, channels, height, width = x.shape

        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {channels}"
            )

        pooled_height, pooled_width = self._get_pooled_size(
            height=height,
            width=width,
        )

        if (
            pooled_height == height
            and pooled_width == width
        ):
            pooled = x
        else:
            pooled = F.adaptive_avg_pool2d(
                x,
                output_size=(
                    pooled_height,
                    pooled_width,
                ),
            )

        # Non-local attention is computed in float32 for numerical
        # stability even when the surrounding network uses AMP/float16.
        pooled_float = pooled.float()

        # [B, C, Hp, Wp] -> [B, N, C]
        positions = pooled_float.flatten(2).transpose(1, 2)

        # Dot-product pairwise affinity:
        # [B, N, C] x [B, C, N] -> [B, N, N]
        affinity = torch.bmm(
            positions,
            positions.transpose(1, 2),
        )

        affinity = torch.softmax(
            affinity,
            dim=-1,
        )

        # [B, N, N] x [B, N, C] -> [B, N, C]
        denoised = torch.bmm(
            affinity,
            positions,
        )

        denoised = denoised.transpose(1, 2).reshape(
            batch_size,
            channels,
            pooled_height,
            pooled_width,
        )

        denoised = denoised.to(dtype=x.dtype)
        denoised = self.output_conv(denoised)

        if (
            pooled_height != height
            or pooled_width != width
        ):
            denoised = F.interpolate(
                denoised,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        return x + denoised


class BoundaryConvBlock(nn.Module):
    """
    Lightweight two-convolution block retained from the E4 boundary-detail path.

    Input / output:
        [B, C_in, H, W] -> [B, C_out, H, W]

    No BatchNorm is used so the normalization setting remains consistent
    with E0-E3.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class BoundaryDetailPath(nn.Module):
    """
    Lightweight boundary-detail pathway retained unchanged from E4.

    Input sources:
        center LOW plane   : [B, 1, 208, 512]
        BEFD edge guidance : [B, 1, 208, 512]

    The two maps are concatenated and processed as a small feature
    pyramid. The branch deliberately remains much lighter than the main
    Fully Dense U-Net context path.

    Outputs:
        b1: [B,  8, 208, 512]
        b2: [B, 16, 104, 256]
        b3: [B, 32,  52, 128]
        b4: [B, 64,  26,  64]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
    ) -> None:
        super().__init__()

        self.block1 = BoundaryConvBlock(
            in_channels=2,
            out_channels=c1,
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.block2 = BoundaryConvBlock(
            in_channels=c1,
            out_channels=c2,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.block3 = BoundaryConvBlock(
            in_channels=c2,
            out_channels=c3,
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Keep the trainable path lightweight: no additional full block at
        # the deepest boundary scale, only a 1x1 channel projection.
        self.project4 = nn.Sequential(
            nn.Conv2d(
                c3,
                c4,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        center: torch.Tensor,
        edge_guidance: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if center.shape != edge_guidance.shape:
            raise ValueError(
                "Boundary path input shape mismatch: "
                f"center={tuple(center.shape)}, "
                f"edge={tuple(edge_guidance.shape)}"
            )

        boundary_input = torch.cat(
            [center, edge_guidance],
            dim=1,
        )

        b1 = self.block1(boundary_input)
        b2 = self.block2(self.pool1(b1))
        b3 = self.block3(self.pool2(b2))
        b4 = self.project4(self.pool3(b3))

        return b1, b2, b3, b4


class GatedBoundaryFusion(nn.Module):
    """
    Fuse a decoder feature with a boundary-detail feature while preserving
    the decoder channel count.

    The boundary feature is first projected to the main feature channel
    count. A learnable sigmoid gate then decides, per channel and per
    pixel, how much boundary information should be injected:

        Bp   = Project(B)
        Gate = sigmoid(Predict([Main, Bp]))
        Out  = Main + Gate * Bp

    Conservative negative bias initialization prevents the detail branch
    from dominating the context branch at the beginning of training.
    """

    def __init__(
        self,
        main_channels: int,
        boundary_channels: int,
        gate_bias_init: float = BOUNDARY_GATE_BIAS_INIT,
    ) -> None:
        super().__init__()

        self.main_channels = int(main_channels)
        self.boundary_channels = int(boundary_channels)

        self.boundary_projection = nn.Conv2d(
            boundary_channels,
            main_channels,
            kernel_size=1,
            bias=True,
        )

        self.gate_predictor = nn.Sequential(
            nn.Conv2d(
                main_channels * 2,
                main_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                main_channels,
                main_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        final_gate_conv = self.gate_predictor[-1]
        if not isinstance(final_gate_conv, nn.Conv2d):
            raise TypeError(
                "Expected final gate predictor layer to be Conv2d."
            )

        nn.init.constant_(
            final_gate_conv.bias,
            float(gate_bias_init),
        )

    def forward(
        self,
        main_feature: torch.Tensor,
        boundary_feature: torch.Tensor,
    ) -> torch.Tensor:
        if main_feature.ndim != 4 or boundary_feature.ndim != 4:
            raise ValueError(
                "GatedBoundaryFusion expects 4D tensors."
            )

        if main_feature.size(1) != self.main_channels:
            raise ValueError(
                f"Expected main channels={self.main_channels}, "
                f"got {main_feature.size(1)}"
            )

        if boundary_feature.size(1) != self.boundary_channels:
            raise ValueError(
                f"Expected boundary channels={self.boundary_channels}, "
                f"got {boundary_feature.size(1)}"
            )

        if boundary_feature.shape[-2:] != main_feature.shape[-2:]:
            boundary_feature = F.interpolate(
                boundary_feature,
                size=main_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        projected_boundary = self.boundary_projection(
            boundary_feature
        )

        gate_input = torch.cat(
            [main_feature, projected_boundary],
            dim=1,
        )

        gate = torch.sigmoid(
            self.gate_predictor(gate_input)
        )

        return (
            main_feature
            + gate * projected_boundary
        )


class AdaptiveFeatureFusion(nn.Module):
    """
    E5 adaptive hierarchical feature fusion module.

    This module is inspired by the SCS-Net AFF principle: adjacent
    hierarchical features should not be merged with an unconditional raw
    concatenation. Instead, the network learns how much information to use
    from each branch according to local spatial context and global channel
    context.

    Inputs:
        decoder_feature: [B, C, H, W]
        skip_feature   : [B, C, H, W]

    Processing:
        1. Project both branches to C channels.
        2. Form a joint context feature from decoder + skip.
        3. Predict branch logits using:
             - local context: spatially varying 1x1 convolutions
             - global context: adaptive average pooling + 1x1 convolutions
        4. Softmax over the two branches:

             w_decoder + w_skip = 1

           for every channel and spatial location.
        5. Return weighted concatenation:

             concat(
                 scale * w_decoder * decoder,
                 scale * w_skip    * skip,
             )

    Output:
        [B, 2C, H, W]

    Controlled-ablation initialization:
        The final local/global logit convolutions are zero-initialized, so
        both weights start at 0.5. With output_scale=2.0, the initial AFF
        output has the same scale as the E4 raw concatenation.

    Important:
        This is an SCS-Net-inspired AFF adaptation for PAM enhancement,
        not a literal reproduction of the original retinal-segmentation
        network implementation.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = AFF_REDUCTION,
        min_hidden_channels: int = AFF_MIN_HIDDEN_CHANNELS,
        output_scale: float = AFF_OUTPUT_SCALE,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(
                f"channels must be > 0, got {channels}"
            )

        if reduction <= 0:
            raise ValueError(
                f"reduction must be > 0, got {reduction}"
            )

        if min_hidden_channels <= 0:
            raise ValueError(
                "min_hidden_channels must be > 0, "
                f"got {min_hidden_channels}"
            )

        if output_scale <= 0:
            raise ValueError(
                f"output_scale must be > 0, got {output_scale}"
            )

        self.channels = int(channels)
        self.output_scale = float(output_scale)

        hidden_channels = max(
            int(min_hidden_channels),
            self.channels // int(reduction),
        )

        # Keep explicit projections so AFF can be extended later without
        # changing the decoder interface. In E5 both inputs already have C
        # channels, but the 1x1 projections remain learnable.
        self.decoder_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=True,
        )

        self.skip_projection = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=1,
            bias=True,
        )

        # Identity initialization makes the two branch projections exactly
        # transparent at startup. Together with equal 0.5/0.5 AFF weights
        # and output_scale=2.0, E5 initially reproduces E4 raw concat:
        #
        #   concat(decoder, skip)
        #
        nn.init.dirac_(self.decoder_projection.weight)
        nn.init.zeros_(self.decoder_projection.bias)
        nn.init.dirac_(self.skip_projection.weight)
        nn.init.zeros_(self.skip_projection.bias)

        # Spatially varying local context.
        self.local_context = nn.Sequential(
            nn.Conv2d(
                self.channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                self.channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

        # Global channel context.
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                self.channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                self.channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

        # Start from exactly equal branch weights. This allows E5 to begin
        # close to E4's raw skip concatenation while still learning adaptive
        # branch selection during training.
        local_final = self.local_context[-1]
        global_final = self.global_context[-1]

        if not isinstance(local_final, nn.Conv2d):
            raise TypeError(
                "Expected final local AFF layer to be Conv2d."
            )

        if not isinstance(global_final, nn.Conv2d):
            raise TypeError(
                "Expected final global AFF layer to be Conv2d."
            )

        nn.init.zeros_(local_final.weight)
        nn.init.zeros_(local_final.bias)
        nn.init.zeros_(global_final.weight)
        nn.init.zeros_(global_final.bias)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        skip_feature: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_feature.ndim != 4 or skip_feature.ndim != 4:
            raise ValueError(
                "AdaptiveFeatureFusion expects 4D tensors."
            )

        if decoder_feature.size(1) != self.channels:
            raise ValueError(
                f"Expected decoder channels={self.channels}, "
                f"got {decoder_feature.size(1)}"
            )

        if skip_feature.size(1) != self.channels:
            raise ValueError(
                f"Expected skip channels={self.channels}, "
                f"got {skip_feature.size(1)}"
            )

        if skip_feature.shape[-2:] != decoder_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        decoder_projected = self.decoder_projection(
            decoder_feature
        )

        skip_projected = self.skip_projection(
            skip_feature
        )

        joint_context = (
            decoder_projected
            + skip_projected
        )

        local_logits = self.local_context(
            joint_context
        )

        global_logits = self.global_context(
            joint_context
        )

        logits = local_logits + global_logits

        batch_size, _, height, width = logits.shape

        # [B, 2C, H, W] -> [B, 2, C, H, W]
        logits = logits.reshape(
            batch_size,
            2,
            self.channels,
            height,
            width,
        )

        weights = torch.softmax(
            logits,
            dim=1,
        )

        decoder_weight = weights[:, 0, :, :, :]
        skip_weight = weights[:, 1, :, :, :]

        weighted_decoder = (
            self.output_scale
            * decoder_weight
            * decoder_projected
        )

        weighted_skip = (
            self.output_scale
            * skip_weight
            * skip_projected
        )

        return torch.cat(
            [weighted_decoder, weighted_skip],
            dim=1,
        )


class DualPathFullyDenseSFABEFDAFFUNet(nn.Module):
    """
    E5 model: Dual-path Fully Dense U-Net + SFA + BEFD + AFF for
    3-adjacent Y-Z PAM image enhancement.

    Controlled ablation relative to E4:
      - The complete E4 dual-path architecture is retained.
      - SCS-Net-inspired AFF is added at Decoder 4, 3, 2, and 1.
      - Each AFF module adaptively weights decoder semantics and the
        corresponding denoised/direct encoder skip feature.
      - Boundary-detail gated fusion from E4 is retained unchanged.
      - Training loss remains L1 and the output remains linear.

    Main context path input:
        [B, 3, 200, 512]
        Three adjacent Y-Z planes: X-1, X, X+1.

    Boundary-detail path input:
        Center LOW plane + BEFD Sobel guidance.

    Main-path BEFD components retained from E3:
        Boundary enhancement:
            e3, e4, bottleneck

        Feature denoising:
            skip1, skip2, skip3

    Dual-path fusion stages:
        Decoder 4 <- b4
        Decoder 3 <- b3
        Decoder 1 <- b1

    Input:
        [B, 3, 200, 512]

    Output:
        [B, 1, 200, 512]
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        out_channels: int = OUTPUT_CHANNELS,
        base_channels: int = BASE_CHANNELS,
        dense_layers_per_block: int = 3,
        sfa_dilation_rates: tuple[int, int, int] = SFA_DILATION_RATES,
        edge_lambda_min: float = BEFD_EDGE_LAMBDA_MIN,
        edge_lambda_max: float = BEFD_EDGE_LAMBDA_MAX,
        edge_alpha: float = BEFD_EDGE_ALPHA,
        edge_beta: float = BEFD_EDGE_BETA,
        denoise_max_positions: int = BEFD_DENOISE_MAX_POSITIONS,
        gate_bias_init: float = BOUNDARY_GATE_BIAS_INIT,
        aff_reduction: int = AFF_REDUCTION,
        aff_min_hidden_channels: int = AFF_MIN_HIDDEN_CHANNELS,
        aff_output_scale: float = AFF_OUTPUT_SCALE,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        if c5 != MAX_CHANNELS:
            raise ValueError(
                f"Channel configuration mismatch: "
                f"bottleneck={c5}, "
                f"MAX_CHANNELS={MAX_CHANNELS}"
            )

        self.dense_layers_per_block = dense_layers_per_block
        self.sfa_dilation_rates = tuple(sfa_dilation_rates)
        self.denoise_max_positions = int(denoise_max_positions)
        self.gate_bias_init = float(gate_bias_init)
        self.aff_reduction = int(aff_reduction)
        self.aff_min_hidden_channels = int(aff_min_hidden_channels)
        self.aff_output_scale = float(aff_output_scale)

        # -------------------------
        # Shared fixed BEFD edge prior
        # -------------------------
        self.boundary_attention = SobelBoundaryAttention(
            lambda_min=edge_lambda_min,
            lambda_max=edge_lambda_max,
            alpha=edge_alpha,
            beta=edge_beta,
        )

        # -------------------------
        # E4 lightweight boundary-detail path retained in E5
        # -------------------------
        self.boundary_path = BoundaryDetailPath(
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
        )

        # -------------------------
        # Main context encoder
        # -------------------------
        self.encoder1 = DenseBlock(
            in_channels,
            c1,
            num_layers=dense_layers_per_block,
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = DenseBlock(
            c1,
            c2,
            num_layers=dense_layers_per_block,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = DenseBlock(
            c2,
            c3,
            num_layers=dense_layers_per_block,
        )
        self.sfa3 = ScaleAwareFeatureAggregation(
            channels=c3,
            dilation_rates=sfa_dilation_rates,
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = DenseBlock(
            c3,
            c4,
            num_layers=dense_layers_per_block,
        )
        self.sfa4 = ScaleAwareFeatureAggregation(
            channels=c4,
            dilation_rates=sfa_dilation_rates,
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # -------------------------
        # Main context bottleneck
        # -------------------------
        self.bottleneck = DenseBlock(
            c4,
            c5,
            num_layers=dense_layers_per_block,
        )
        self.sfa_bottleneck = ScaleAwareFeatureAggregation(
            channels=c5,
            dilation_rates=sfa_dilation_rates,
        )

        # -------------------------
        # E3 BEFD feature denoising retained
        # -------------------------
        self.denoise1 = FeatureDenoisingBlock(
            channels=c1,
            max_positions=denoise_max_positions,
        )
        self.denoise2 = FeatureDenoisingBlock(
            channels=c2,
            max_positions=denoise_max_positions,
        )
        self.denoise3 = FeatureDenoisingBlock(
            channels=c3,
            max_positions=denoise_max_positions,
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.upconv4 = nn.ConvTranspose2d(
            c5,
            c4,
            kernel_size=2,
            stride=2,
        )
        self.boundary_fusion4 = GatedBoundaryFusion(
            main_channels=c4,
            boundary_channels=c4,
            gate_bias_init=gate_bias_init,
        )
        self.aff4 = AdaptiveFeatureFusion(
            channels=c4,
            reduction=aff_reduction,
            min_hidden_channels=aff_min_hidden_channels,
            output_scale=aff_output_scale,
        )
        self.decoder4 = DenseBlock(
            c4 + c4,
            c4,
            num_layers=dense_layers_per_block,
        )

        self.upconv3 = nn.ConvTranspose2d(
            c4,
            c3,
            kernel_size=2,
            stride=2,
        )
        self.boundary_fusion3 = GatedBoundaryFusion(
            main_channels=c3,
            boundary_channels=c3,
            gate_bias_init=gate_bias_init,
        )
        self.aff3 = AdaptiveFeatureFusion(
            channels=c3,
            reduction=aff_reduction,
            min_hidden_channels=aff_min_hidden_channels,
            output_scale=aff_output_scale,
        )
        self.decoder3 = DenseBlock(
            c3 + c3,
            c3,
            num_layers=dense_layers_per_block,
        )

        self.upconv2 = nn.ConvTranspose2d(
            c3,
            c2,
            kernel_size=2,
            stride=2,
        )
        self.aff2 = AdaptiveFeatureFusion(
            channels=c2,
            reduction=aff_reduction,
            min_hidden_channels=aff_min_hidden_channels,
            output_scale=aff_output_scale,
        )
        self.decoder2 = DenseBlock(
            c2 + c2,
            c2,
            num_layers=dense_layers_per_block,
        )

        self.upconv1 = nn.ConvTranspose2d(
            c2,
            c1,
            kernel_size=2,
            stride=2,
        )
        self.boundary_fusion1 = GatedBoundaryFusion(
            main_channels=c1,
            boundary_channels=c1,
            gate_bias_init=gate_bias_init,
        )
        self.aff1 = AdaptiveFeatureFusion(
            channels=c1,
            reduction=aff_reduction,
            min_hidden_channels=aff_min_hidden_channels,
            output_scale=aff_output_scale,
        )
        self.decoder1 = DenseBlock(
            c1 + c1,
            c1,
            num_layers=dense_layers_per_block,
        )

        # Linear output, unchanged from E0-E3.
        self.output_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    @staticmethod
    def _resize_attention(
        attention: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return F.interpolate(
            attention,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # [B, 3, 200, 512]
        # -> [B, 3, 208, 512]
        x = F.pad(
            x,
            pad=(
                0,
                0,
                PAD_TOP,
                PAD_BOTTOM,
            ),
            mode="reflect",
        )

        # Center LOW plane corresponding to HIGH[x, :, :].
        center = x[:, 1:2, :, :]

        # Shared fixed BEFD Sobel-derived attention.
        # [B, 1, 208, 512]
        edge_attention = self.boundary_attention(
            center
        )

        # Convert multiplicative attention into a non-negative guidance
        # signal for the detail branch: neutral attention 1 -> guidance 0.
        edge_guidance = (
            edge_attention - 1.0
        ).clamp_min(0.0)

        # ----------------------------------------------------
        # Path B: lightweight boundary-detail feature pyramid
        # ----------------------------------------------------
        b1, b2, b3, b4 = self.boundary_path(
            center=center,
            edge_guidance=edge_guidance,
        )

        # ----------------------------------------------------
        # Path A: E3 main context path retained unchanged
        # ----------------------------------------------------
        e1 = self.encoder1(x)

        e2 = self.encoder2(
            self.pool1(e1)
        )

        e3 = self.encoder3(
            self.pool2(e2)
        )
        e3 = self.sfa3(e3)
        attention3 = self._resize_attention(
            edge_attention,
            e3,
        )
        e3 = e3 * attention3

        e4 = self.encoder4(
            self.pool3(e3)
        )
        e4 = self.sfa4(e4)
        attention4 = self._resize_attention(
            edge_attention,
            e4,
        )
        e4 = e4 * attention4

        bottleneck = self.bottleneck(
            self.pool4(e4)
        )
        bottleneck = self.sfa_bottleneck(
            bottleneck
        )
        attention_bottleneck = self._resize_attention(
            edge_attention,
            bottleneck,
        )
        bottleneck = (
            bottleneck * attention_bottleneck
        )

        # E3 BEFD non-local denoising on the first three skips.
        skip1 = self.denoise1(e1)
        skip2 = self.denoise2(e2)
        skip3 = self.denoise3(e3)

        # ----------------------------------------------------
        # Decoder with E4 gated dual-path fusion + E5 AFF
        # ----------------------------------------------------
        d4 = self.upconv4(bottleneck)
        d4 = self.boundary_fusion4(
            main_feature=d4,
            boundary_feature=b4,
        )
        fused4 = self.aff4(
            decoder_feature=d4,
            skip_feature=e4,
        )
        d4 = self.decoder4(fused4)

        d3 = self.upconv3(d4)
        d3 = self.boundary_fusion3(
            main_feature=d3,
            boundary_feature=b3,
        )
        fused3 = self.aff3(
            decoder_feature=d3,
            skip_feature=skip3,
        )
        d3 = self.decoder3(fused3)

        d2 = self.upconv2(d3)
        fused2 = self.aff2(
            decoder_feature=d2,
            skip_feature=skip2,
        )
        d2 = self.decoder2(fused2)

        d1 = self.upconv1(d2)
        d1 = self.boundary_fusion1(
            main_feature=d1,
            boundary_feature=b1,
        )
        fused1 = self.aff1(
            decoder_feature=d1,
            skip_feature=skip1,
        )
        d1 = self.decoder1(fused1)

        output = self.output_conv(d1)

        # [B, 1, 208, 512]
        # -> [B, 1, 200, 512]
        output = output[
            :,
            :,
            PAD_TOP:-PAD_BOTTOM,
            :,
        ]

        return output


# ============================================================
# DataLoader
# ============================================================


def create_train_loader() -> tuple[
    PAMAdjacentYZDataset,
    EpochRandomXSliceSampler,
    DataLoader,
]:
    train_dataset = PAMAdjacentYZDataset(
        low_dir=TRAIN_LOW_DIR,
        high_dir=TRAIN_HIGH_DIR,
    )

    train_sampler = EpochRandomXSliceSampler(
        dataset=train_dataset,
        slices_per_volume=(
            SLICES_PER_VOLUME_PER_EPOCH
        ),
        seed=SEED,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": train_dataset,
        "batch_size": BATCH_SIZE,
        "sampler": train_sampler,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "drop_last": False,
        "persistent_workers": PERSISTENT_WORKERS,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = (
            PREFETCH_FACTOR
        )

    train_loader = DataLoader(
        **loader_kwargs
    )

    return (
        train_dataset,
        train_sampler,
        train_loader,
    )


# ============================================================
# Verification / helpers
# ============================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@torch.no_grad()
def verify_model_shape(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    """
    Verify one batch before training.
    """

    batch = next(iter(train_loader))

    inputs = batch["input"].to(
        device=device,
        non_blocking=True,
    )

    targets = batch["target"].to(
        device=device,
        non_blocking=True,
    )

    if (
        USE_CHANNELS_LAST
        and device.type == "cuda"
    ):
        inputs = inputs.contiguous(
            memory_format=torch.channels_last
        )

    model.eval()

    with torch.cuda.amp.autocast(
        enabled=amp_enabled
    ):
        outputs = model(inputs)

    print("\n" + "=" * 70)
    print("MODEL SHAPE VERIFICATION")
    print("=" * 70)

    print(
        f"Input shape : {inputs.shape}"
    )
    print(
        f"Output shape: {outputs.shape}"
    )
    print(
        f"Target shape: {targets.shape}"
    )

    if outputs.shape != targets.shape:
        raise ValueError(
            "\nModel output and target shape mismatch.\n"
            f"Output: {outputs.shape}\n"
            f"Target: {targets.shape}"
        )

    print(
        "\nModel output shape is valid."
    )


def print_cuda_memory(
    label: str,
) -> None:
    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(
        f"\nCUDA Memory [{label}]"
    )
    print(
        f"  Allocated : {allocated:.2f} GB"
    )
    print(
        f"  Reserved  : {reserved:.2f} GB"
    )
    print(
        f"  Peak      : {peak:.2f} GB"
    )


# ============================================================
# Training
# ============================================================


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    amp_enabled: bool,
) -> float:
    model.train()

    running_loss = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )

    total_samples = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=(
            f"Epoch {epoch:03d}/"
            f"{total_epochs:03d}"
        ),
        leave=True,
        mininterval=0.5,
    )

    for step, batch in progress_bar:
        inputs = batch["input"].to(
            device=device,
            non_blocking=True,
        )

        targets = batch["target"].to(
            device=device,
            non_blocking=True,
        )

        if (
            USE_CHANNELS_LAST
            and device.type == "cuda"
        ):
            inputs = inputs.contiguous(
                memory_format=torch.channels_last
            )

        current_batch_size = inputs.size(0)

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):
            predictions = model(inputs)

            # Raw linear prediction.
            # No clamp or output activation in training.
            raw_loss = criterion(
                predictions,
                targets,
            )

            loss_for_backward = (
                raw_loss
                / GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(
            loss_for_backward
        ).backward()

        is_accumulation_boundary = (
            (step + 1)
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        )

        is_last_batch = (
            step + 1
            == len(train_loader)
        )

        if (
            is_accumulation_boundary
            or is_last_batch
        ):
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        running_loss += (
            raw_loss.detach().double()
            * current_batch_size
        )

        total_samples += current_batch_size

        if (
            (step + 1) % LOG_EVERY == 0
            or is_last_batch
        ):
            mean_loss = (
                running_loss
                / total_samples
            ).item()

            progress_bar.set_postfix(
                {
                    "L1": (
                        f"{mean_loss:.6f}"
                    )
                }
            )

    return (
        running_loss
        / total_samples
    ).item()


# ============================================================
# Train MAP visualization
# ============================================================


@torch.no_grad()
def save_epoch_map(
    model: nn.Module,
    train_dataset: PAMAdjacentYZDataset,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    volume_index: int = MAP_VOLUME_INDEX,
) -> None:
    """
    Uses one fixed Train volume.

    The full 200 X positions are inferred:

        200 Y-Z predictions
        -> [X, 1, Y, Z]
        -> channel removal
        -> [X, Y, Z]
        -> max over Z
        -> X-Y MAP

    This is only visualization.
    It is not validation or test.
    """

    if (
        volume_index < 0
        or volume_index
        >= train_dataset.num_volumes
    ):
        raise IndexError(
            f"Invalid MAP volume index: "
            f"{volume_index}. "
            f"Valid range: 0 ~ "
            f"{train_dataset.num_volumes - 1}"
        )

    model.eval()

    start_index = (
        volume_index
        * train_dataset.samples_per_volume
    )

    end_index = (
        start_index
        + train_dataset.samples_per_volume
    )

    volume_subset = Subset(
        train_dataset,
        list(
            range(
                start_index,
                end_index,
            )
        ),
    )

    volume_loader = DataLoader(
        dataset=volume_subset,
        batch_size=MAP_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    low_slices: list[torch.Tensor] = []
    prediction_slices: list[torch.Tensor] = []
    target_slices: list[torch.Tensor] = []

    volume_key = None

    for batch in volume_loader:
        inputs = batch["input"].to(
            device=device,
            non_blocking=True,
        )

        targets = batch["target"]

        if (
            USE_CHANNELS_LAST
            and device.type == "cuda"
        ):
            inputs = inputs.contiguous(
                memory_format=torch.channels_last
            )

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):
            predictions = model(inputs)

        # Visualization only.
        predictions_for_map = (
            predictions.clamp_min(0.0)
        )

        # Center LOW channel:
        # [B, 3, Y, Z]
        # -> [B, 1, Y, Z]
        current_low = inputs[
            :,
            1:2,
            :,
            :,
        ]

        low_slices.append(
            current_low
            .detach()
            .float()
            .cpu()
        )

        prediction_slices.append(
            predictions_for_map
            .detach()
            .float()
            .cpu()
        )

        target_slices.append(
            targets
            .detach()
            .float()
            .cpu()
        )

        if volume_key is None:
            volume_key = (
                batch["volume_key"][0]
            )

    low_slices_tensor = torch.cat(
        low_slices,
        dim=0,
    )

    prediction_slices_tensor = torch.cat(
        prediction_slices,
        dim=0,
    )

    target_slices_tensor = torch.cat(
        target_slices,
        dim=0,
    )

    expected_shape = (
        X_SIZE,
        1,
        Y_SIZE,
        Z_SIZE,
    )

    for label, tensor in (
        ("LOW", low_slices_tensor),
        (
            "Prediction",
            prediction_slices_tensor,
        ),
        (
            "Target",
            target_slices_tensor,
        ),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"\n{label} reconstruction "
                f"shape mismatch.\n"
                f"Actual   : "
                f"{tuple(tensor.shape)}\n"
                f"Expected : "
                f"{expected_shape}"
            )

    # [X, 1, Y, Z] -> [X, Y, Z]
    #
    # No transpose is required.
    low_volume = (
        low_slices_tensor[:, 0, :, :]
        .numpy()
    )

    pred_volume = (
        prediction_slices_tensor[:, 0, :, :]
        .numpy()
    )

    target_volume = (
        target_slices_tensor[:, 0, :, :]
        .numpy()
    )

    for label, volume in (
        ("LOW", low_volume),
        ("Prediction", pred_volume),
        ("Target", target_volume),
    ):
        if tuple(volume.shape) != HIGH_VOLUME_SHAPE:
            raise ValueError(
                f"\n{label} full-volume "
                f"shape mismatch.\n"
                f"Actual   : "
                f"{tuple(volume.shape)}\n"
                f"Expected : "
                f"{HIGH_VOLUME_SHAPE}"
            )

    # X-Y MAP:
    # [X, Y, Z] -> max over Z -> [X, Y]
    low_map = np.max(
        low_volume,
        axis=2,
    )

    pred_map = np.max(
        pred_volume,
        axis=2,
    )

    target_map = np.max(
        target_volume,
        axis=2,
    )

    absolute_error_map = np.abs(
        pred_map
        - target_map
    )

    display_vmin = 0.0

    display_vmax = float(
        np.percentile(
            target_map,
            99.5,
        )
    )

    if (
        not np.isfinite(display_vmax)
        or display_vmax <= display_vmin
    ):
        display_vmax = float(
            target_map.max()
        )

    if display_vmax <= display_vmin:
        display_vmax = 1.0

    error_vmax = float(
        np.percentile(
            absolute_error_map,
            99.5,
        )
    )

    if (
        not np.isfinite(error_vmax)
        or error_vmax <= 0
    ):
        error_vmax = 1.0

    figure, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(20, 5),
    )

    image_0 = axes[0].imshow(
        low_map,
        cmap="hot",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[0].set_title("LOW MAP")
    axes[0].axis("off")
    figure.colorbar(
        image_0,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )

    image_1 = axes[1].imshow(
        pred_map,
        cmap="hot",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[1].set_title(
        f"Prediction MAP\nEpoch {epoch}"
    )
    axes[1].axis("off")
    figure.colorbar(
        image_1,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    image_2 = axes[2].imshow(
        target_map,
        cmap="hot",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[2].set_title("HIGH MAP")
    axes[2].axis("off")
    figure.colorbar(
        image_2,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    image_3 = axes[3].imshow(
        absolute_error_map,
        cmap="viridis",
        vmin=0.0,
        vmax=error_vmax,
    )
    axes[3].set_title(
        "Absolute Error"
    )
    axes[3].axis("off")
    figure.colorbar(
        image_3,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
    )

    figure.suptitle(
        (
            "Dual-Path Fully Dense + SFA + BEFD + AFF U-Net Train MAP | "
            f"{volume_key} | "
            f"Epoch {epoch}"
        ),
        fontsize=14,
    )

    figure.tight_layout()

    output_path = (
        TRAIN_MAP_DIR
        / f"epoch_{epoch:03d}_map.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved Train MAP: "
        f"{output_path}"
    )

    model.train()


# ============================================================
# Saving utilities
# ============================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    train_loss: float,
    samples_per_epoch: int,
    path: Path,
) -> None:
    checkpoint_data = {
        "epoch": epoch,

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "scaler_state_dict": (
            scaler.state_dict()
        ),

        "train_loss": train_loss,

        "model_name": "DualPathFullyDenseSFABEFDAFFUNet",

        "experiment": "E5",

        "training_plane": "YZ",

        "adjacent_axis": "X",

        "low_data_shape_order": (
            "[X, C, Y, Z]"
        ),

        "high_data_shape_order": (
            "[X, Y, Z]"
        ),

        "high_target_definition": (
            "HIGH[x, :, :]"
        ),

        "input_channels": INPUT_CHANNELS,

        "output_channels": OUTPUT_CHANNELS,

        "base_channels": BASE_CHANNELS,

        "max_channels": MAX_CHANNELS,

        "dense_layers_per_block": 3,

        "dense_growth_rule": "max(4, out_channels // 2)",

        "dense_transition": "1x1 Conv + ReLU",

        "sfa_enabled": True,

        "sfa_stages": list(SFA_STAGES),

        "sfa_dilation_rates": list(SFA_DILATION_RATES),

        "sfa_adjacent_pairs": ["1-3", "3-5"],

        "sfa_dynamic_selection": (
            "per-channel per-pixel softmax over adjacent scales"
        ),

        "sfa_residual_fusion": True,

        "befd_enabled": True,

        "befd_boundary_stages": list(BEFD_BOUNDARY_STAGES),

        "befd_boundary_source": "center LOW channel (X)",

        "befd_edge_detector": "fixed Sobel x/y magnitude",

        "befd_edge_lambda_min": BEFD_EDGE_LAMBDA_MIN,

        "befd_edge_lambda_max": BEFD_EDGE_LAMBDA_MAX,

        "befd_edge_alpha": BEFD_EDGE_ALPHA,

        "befd_edge_beta": BEFD_EDGE_BETA,

        "befd_feature_denoising_stages": list(
            BEFD_DENOISE_SKIP_STAGES
        ),

        "befd_denoising_type": (
            "pooled dot-product non-local attention + "
            "1x1 Conv + residual"
        ),

        "befd_denoise_max_positions": (
            BEFD_DENOISE_MAX_POSITIONS
        ),

        "dual_path_enabled": True,

        "dual_path_main_path": (
            "3-adjacent Fully Dense U-Net + SFA + BEFD"
        ),

        "dual_path_boundary_input": (
            "center LOW + BEFD Sobel guidance"
        ),

        "dual_path_boundary_channels": list(
            BOUNDARY_PATH_CHANNELS
        ),

        "dual_path_boundary_fusion_stages": list(
            BOUNDARY_FUSION_STAGES
        ),

        "dual_path_fusion_type": (
            "per-channel per-pixel sigmoid gated residual injection"
        ),

        "dual_path_gate_bias_init": (
            BOUNDARY_GATE_BIAS_INIT
        ),

        "aff_enabled": True,

        "aff_stages": list(AFF_STAGES),

        "aff_type": (
            "SCS-Net-inspired local+global adaptive hierarchical fusion"
        ),

        "aff_weighting": (
            "two-branch per-channel per-pixel softmax"
        ),

        "aff_reduction": AFF_REDUCTION,

        "aff_min_hidden_channels": AFF_MIN_HIDDEN_CHANNELS,

        "aff_output_scale": AFF_OUTPUT_SCALE,

        "aff_output": (
            "weighted concatenation preserving 2C decoder input"
        ),

        "aff_initialization": (
            "zero final logits -> equal 0.5/0.5 branch weights"
        ),

        "physical_batch_size": BATCH_SIZE,

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),

        "effective_batch_size": (
            BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),

        "slices_per_volume_per_epoch": (
            SLICES_PER_VOLUME_PER_EPOCH
        ),

        "samples_per_epoch": (
            samples_per_epoch
        ),

        "use_amp": USE_AMP,

        "use_channels_last": (
            USE_CHANNELS_LAST
        ),

        "final_output_activation": (
            "None (Linear)"
        ),

        "loss": "L1Loss",

        "optimizer": "Adam",

        "learning_rate": LEARNING_RATE,

        "map_save_every": MAP_SAVE_EVERY,
    }

    torch.save(
        checkpoint_data,
        path,
    )


def save_training_history(
    history: list[dict[str, float]],
) -> None:
    with HISTORY_CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "epoch",
                "train_l1_loss",
                "epoch_time_seconds",
                "peak_gpu_memory_gb",
                "samples_this_epoch",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


def save_loss_curve(
    history: list[dict[str, float]],
) -> None:
    epochs = [
        row["epoch"]
        for row in history
    ]

    losses = [
        row["train_l1_loss"]
        for row in history
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        epochs,
        losses,
        marker="o",
        markersize=3,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Train L1 Loss")

    plt.title(
        "Dual-Path Fully Dense + SFA + BEFD + AFF U-Net Training Loss"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        LOSS_CURVE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Optimizer
# ============================================================


def create_optimizer(
    model: nn.Module,
    device: torch.device,
) -> torch.optim.Optimizer:
    """
    Try fused Adam on CUDA.
    Fall back to standard Adam if unsupported.
    """

    base_kwargs = {
        "params": model.parameters(),
        "lr": LEARNING_RATE,
        "betas": (0.9, 0.999),
    }

    if device.type == "cuda":
        try:
            optimizer = torch.optim.Adam(
                **base_kwargs,
                fused=True,
            )

            print(
                "  Adam implementation    : fused"
            )

            return optimizer

        except (
            TypeError,
            RuntimeError,
        ):
            pass

    print(
        "  Adam implementation    : standard"
    )

    return torch.optim.Adam(
        **base_kwargs
    )


# ============================================================
# Full training
# ============================================================


def train() -> None:
    set_seed(SEED)
    configure_cuda()
    create_output_directories()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    print("\n" + "=" * 70)

    print(
        "PAM 3-ADJACENT Y-Z DUAL-PATH "
        "FULLY DENSE + SFA + BEFD + AFF U-NET TRAINING"
    )

    print(
        "TRAIN-ONLY | "
        "PRE-GENERATED 3-ADJACENT LOW DATA"
    )

    print("=" * 70)

    print(
        f"Device                  : "
        f"{device}"
    )

    if torch.cuda.is_available():
        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        total_gpu_memory = (
            torch.cuda
            .get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU                     : "
            f"{gpu_name}"
        )

        print(
            f"Total GPU memory        : "
            f"{total_gpu_memory:.2f} GB"
        )

    (
        train_dataset,
        train_sampler,
        train_loader,
    ) = create_train_loader()

    samples_per_epoch = len(
        train_sampler
    )

    print("\nData")

    print(
        f"  Train LOW directory    : "
        f"{TRAIN_LOW_DIR}"
    )

    print(
        f"  Train HIGH directory   : "
        f"{TRAIN_HIGH_DIR}"
    )

    print(
        f"  Train volumes          : "
        f"{train_dataset.num_volumes}"
    )

    print(
        f"  Full train samples     : "
        f"{len(train_dataset):,}"
    )

    print(
        f"  X positions/vol/epoch : "
        f"{SLICES_PER_VOLUME_PER_EPOCH} / "
        f"{train_dataset.samples_per_volume}"
    )

    print(
        f"  Samples this epoch     : "
        f"{samples_per_epoch:,}"
    )

    print(
        f"  Physical batch size    : "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Batches/epoch          : "
        f"{len(train_loader):,}"
    )

    print(
        f"  Num workers            : "
        f"{NUM_WORKERS}"
    )

    print(
        f"  Persistent workers     : "
        f"{PERSISTENT_WORKERS}"
    )

    print(
        f"  Prefetch factor        : "
        f"{PREFETCH_FACTOR if NUM_WORKERS > 0 else 'N/A'}"
    )

    model = DualPathFullyDenseSFABEFDAFFUNet(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
    ).to(device)

    if (
        USE_CHANNELS_LAST
        and device.type == "cuda"
    ):
        model = model.to(
            memory_format=torch.channels_last
        )

    parameter_count = (
        count_trainable_parameters(model)
    )

    print("\nModel")

    print(
        "  Architecture           : "
        "Dual-Path Fully Dense U-Net + SFA + BEFD + AFF (E5)"
    )

    print(
        "  Training plane         : "
        "Y-Z"
    )

    print(
        "  Adjacent axis          : "
        "X"
    )

    print(
        "  Channels               : "
        "3-8-16-32-64-128-64-32-16-8-1"
    )

    print(
        f"  Parameters             : "
        f"{parameter_count:,}"
    )

    print(
        "  Dense layers/block     : 3"
    )

    print(
        "  SFA stages             : "
        "Encoder3, Encoder4, Bottleneck"
    )

    print(
        "  SFA dilation rates     : "
        f"{SFA_DILATION_RATES}"
    )

    print(
        "  SFA adjacent fusion    : "
        "(1,3) and (3,5), dynamic softmax"
    )

    print(
        "  BEFD boundary stages   : "
        "Encoder3, Encoder4, Bottleneck"
    )

    print(
        "  BEFD edge source       : "
        "Center LOW channel + fixed Sobel"
    )

    print(
        "  BEFD denoised skips    : "
        "Encoder1, Encoder2, Encoder3"
    )

    print(
        "  Dual-path              : Enabled"
    )

    print(
        "  Boundary path input     : "
        "Center LOW + Sobel guidance"
    )

    print(
        "  Boundary path channels  : "
        f"{BOUNDARY_PATH_CHANNELS}"
    )

    print(
        "  Boundary fusion stages  : "
        "Decoder4, Decoder3, Decoder1"
    )

    print(
        "  Boundary fusion type    : "
        "Gated residual injection"
    )

    print(
        "  FD non-local positions : "
        f"<= {BEFD_DENOISE_MAX_POSITIONS}"
    )

    print(
        "  AFF                    : Enabled"
    )

    print(
        "  AFF stages             : "
        "Decoder4, Decoder3, Decoder2, Decoder1"
    )

    print(
        "  AFF context            : "
        "Local spatial + global channel"
    )

    print(
        "  AFF weighting          : "
        "2-branch per-channel per-pixel softmax"
    )

    print(
        "  AFF output             : "
        "Weighted concatenation (2C)"
    )

    print(
        "  AFF initial behavior   : "
        "Equivalent scale to raw E4 concatenation"
    )

    print(
        "  BatchNorm              : None"
    )

    print(
        "  Final output activation: "
        "None (Linear)"
    )

    print(
        f"  Channels last          : "
        f"{USE_CHANNELS_LAST}"
    )

    criterion = nn.L1Loss()

    print("\nOptimizer")

    optimizer = create_optimizer(
        model,
        device,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled
    )

    print("\nTraining configuration")

    print(
        f"  Epochs                 : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"  Learning rate          : "
        f"{LEARNING_RATE}"
    )

    print(
        "  Loss                   : "
        "L1 Loss"
    )

    print(
        "  Optimizer              : "
        "Adam"
    )

    print(
        f"  AMP                    : "
        f"{amp_enabled}"
    )

    print(
        "  Validation             : None"
    )

    print(
        "  Test                   : Not used"
    )

    print(
        f"  MAP save every         : "
        f"{MAP_SAVE_EVERY} epochs"
    )

    print(
        f"  MAP volume index       : "
        f"{MAP_VOLUME_INDEX}"
    )

    # --------------------------------------------------------
    # Shape verification
    # --------------------------------------------------------

    train_sampler.set_epoch(0)

    verify_model_shape(
        model=model,
        train_loader=train_loader,
        device=device,
        amp_enabled=amp_enabled,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print_cuda_memory(
        "Before Training"
    )

    # --------------------------------------------------------
    # Optional initial Train MAP
    # --------------------------------------------------------

    if SAVE_INITIAL_MAP:
        print(
            "\nGenerating initial "
            "Epoch 0 Train MAP..."
        )

        save_epoch_map(
            model=model,
            train_dataset=train_dataset,
            device=device,
            epoch=0,
            amp_enabled=amp_enabled,
            volume_index=MAP_VOLUME_INDEX,
        )

    history: list[dict[str, float]] = []

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        # Different deterministic random X subset
        # for every epoch.
        train_sampler.set_epoch(epoch)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        epoch_start_time = (
            time.perf_counter()
        )

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=NUM_EPOCHS,
            amp_enabled=amp_enabled,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        if torch.cuda.is_available():
            peak_gpu_memory_gb = (
                torch.cuda
                .max_memory_allocated()
                / (1024 ** 3)
            )
        else:
            peak_gpu_memory_gb = 0.0

        history.append(
            {
                "epoch": epoch,
                "train_l1_loss": train_loss,
                "epoch_time_seconds": (
                    epoch_time
                ),
                "peak_gpu_memory_gb": (
                    peak_gpu_memory_gb
                ),
                "samples_this_epoch": (
                    samples_per_epoch
                ),
            }
        )

        print(
            f"\nEpoch "
            f"{epoch:03d}/{NUM_EPOCHS:03d}"
        )

        print(
            f"Train L1 Loss   : "
            f"{train_loss:.8f}"
        )

        print(
            f"Epoch Time      : "
            f"{epoch_time:.2f} sec"
        )

        print(
            f"Peak GPU Memory : "
            f"{peak_gpu_memory_gb:.2f} GB"
        )

        print(
            f"Samples trained : "
            f"{samples_per_epoch:,}"
        )

        # ----------------------------------------------------
        # Train MAP visualization every N epochs.
        # This is not validation or test.
        # ----------------------------------------------------

        if epoch % MAP_SAVE_EVERY == 0:
            print(
                f"\nGenerating full-volume "
                f"Train MAP for Epoch {epoch}..."
            )

            save_epoch_map(
                model=model,
                train_dataset=train_dataset,
                device=device,
                epoch=epoch,
                amp_enabled=amp_enabled,
                volume_index=MAP_VOLUME_INDEX,
            )

        # ----------------------------------------------------
        # Latest checkpoint every epoch
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            samples_per_epoch=samples_per_epoch,
            path=LATEST_MODEL_PATH,
        )

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if epoch % CHECKPOINT_EVERY == 0:
            checkpoint_path = (
                CHECKPOINT_DIR
                / f"epoch_{epoch:03d}.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                samples_per_epoch=samples_per_epoch,
                path=checkpoint_path,
            )

            print(
                f"Saved checkpoint: "
                f"{checkpoint_path}"
            )

        save_training_history(history)
        save_loss_curve(history)

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    final_train_loss = (
        history[-1]["train_l1_loss"]
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=NUM_EPOCHS,
        train_loss=final_train_loss,
        samples_per_epoch=samples_per_epoch,
        path=FINAL_MODEL_PATH,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal model:\n  "
        f"{FINAL_MODEL_PATH}"
    )

    print(
        f"\nLatest model:\n  "
        f"{LATEST_MODEL_PATH}"
    )

    print(
        f"\nTraining MAP images:\n  "
        f"{TRAIN_MAP_DIR}"
    )

    print(
        f"\nTraining history:\n  "
        f"{HISTORY_CSV_PATH}"
    )

    print(
        f"\nLoss curve:\n  "
        f"{LOSS_CURVE_PATH}"
    )

    train_dataset.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    train()
