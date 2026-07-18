# -*- coding: utf-8 -*-
"""
.1 PAM intensity-aware image enhancement training script.

Completely new architecture (not an extension of E5):

    3-adjacent Y-Z LOW input [B, 3, 200, 512]
        |
        +--> Main restoration encoder-decoder
        |       - residual restoration blocks
        |       - PixelShuffle decoder
        |       - large-receptive-field bottleneck
        |
        +--> Full-resolution edge branch
        |       - fixed Sobel gradient
        |       - fixed Laplacian response
        |
        +--> Haar-wavelet high-frequency branch
                - LH / HL / HH bands
                - high-frequency energy guidance

    Main residual + gated detail residual
        -> Stage-1 coarse prediction

    Stage-1 coarse prediction + original center LOW + edge/frequency guidance
        -> full-resolution detail refiner
        -> Stage-2 residual correction
        -> final prediction

Residual reconstruction:

    coarse = center + alpha * R_main + beta * G_detail * R_detail
    final  = coarse + gamma * G_refine * R_refine

Design goals:
    1. Preserve real vessel structures already present in LOW.
    2. Reduce over-smoothing / image blur.
    3. Recover thin vessels and boundaries through explicit edge/frequency paths.
    4. Preserve PAM intensity and bright-vessel amplitude through target-weighted loss.
    5. Prevent global darkness / contrast compression with intensity-statistics loss.
    6. Limit hallucination with conservative gated residual injection.
    7. Keep the model practical for an RTX 3060 Ti.

Current project data specification:
    LOW  : ./data/3d_Train/LOW/*.bin
           float32 + 48-byte header
           [X, C, Y, Z] = [200, 3, 200, 512]

    HIGH : ./data/norm_Train/HIGH/*.bin
           float32 + 48-byte header
           [X, Y, Z] = [200, 200, 512]

This script trains only. No validation/test split is used.
"""

from __future__ import annotations

import csv
import math
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
# 1. Configuration
# ============================================================

SEED = 42


class DataConfig:
    def __init__(self) -> None:
        self.data_root = Path("./data")
        self.train_low_dir = Path("./data/3d_Train/LOW")
        self.train_high_dir = Path("./data/norm_Train/HIGH")

        self.header_size = 48
        self.dtype = np.float32

        self.high_volume_shape = (200, 200, 512)      # [X,Y,Z]
        self.adjacent_low_shape = (200, 3, 200, 512)  # [X,C,Y,Z]

        self.input_channels = 3
        self.output_channels = 1


class ModelConfig:
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        max_channels: int = 128,
        encoder_blocks: tuple[int, int, int] = (2, 2, 3),
        bottleneck_blocks: int = 4,
        decoder_blocks: tuple[int, int, int] = (2, 2, 2),
        edge_channels: int = 16,
        wavelet_channels: int = 16,
        detail_channels: int = 16,
        refiner_channels: int = 16,
        refiner_dilations: tuple[int, ...] = (1, 2, 3, 1, 2, 1),
        alpha_init: float = 0.20,
        beta_init: float = 0.10,
        gamma_init: float = 0.20,
        detail_gate_bias_init: float = -2.0,
        refine_gate_bias_init: float = -1.0,
        use_edge_branch: bool = True,
        use_wavelet_branch: bool = True,
        use_stage2: bool = True,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # RTX 3060 Ti-oriented channel width.
        self.base_channels = int(base_channels)
        self.max_channels = int(max_channels)

        # Main restoration branch blocks per level.
        self.encoder_blocks = tuple(int(v) for v in encoder_blocks)
        self.bottleneck_blocks = int(bottleneck_blocks)
        self.decoder_blocks = tuple(int(v) for v in decoder_blocks)

        # Edge/frequency detail branches.
        self.edge_channels = int(edge_channels)
        self.wavelet_channels = int(wavelet_channels)
        self.detail_channels = int(detail_channels)

        # Stage-2 full-resolution refinement.
        self.refiner_channels = int(refiner_channels)
        self.refiner_dilations = tuple(int(v) for v in refiner_dilations)

        # Conservative residual contribution at initialization.
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self.gamma_init = float(gamma_init)

        # Negative gate bias -> sigmoid gate starts small.
        self.detail_gate_bias_init = float(detail_gate_bias_init)
        self.refine_gate_bias_init = float(refine_gate_bias_init)

        # Ablation controls. Keep all True for full .
        self.use_edge_branch = bool(use_edge_branch)
        self.use_wavelet_branch = bool(use_wavelet_branch)
        self.use_stage2 = bool(use_stage2)


class TrainConfig:
    def __init__(self) -> None:
        self.num_epochs = 50
        self.learning_rate = 2e-4
        self.weight_decay = 1e-6

        # 130 volumes x 32 X positions = 4,160 samples/epoch.
        self.slices_per_volume_per_epoch = 32

        # Conservative default for RTX 3060 Ti 8 GB.
        self.batch_size = 4
        self.gradient_accumulation_steps = 2

        self.num_workers = 4
        self.prefetch_factor = 2
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = True

        self.use_amp = True
        self.use_channels_last = True

        self.log_every = 20

        # ------------------------------------------------------------
        # .1 intensity-aware composite loss
        #
        # Final-stage weights are normalized to 1.0:
        #   20% L1
        #   45% SSIM
        #   10% boundary gradient
        #   20% target-weighted intensity L1
        #    5% weighted intensity-statistics loss
        #
        # Compared with , SSIM is still the largest single term, but the
        # objective now provides substantially stronger supervision for PAM
        # signal amplitude and bright-vessel intensity.
        # ------------------------------------------------------------
        self.final_l1_weight = 0.20
        self.final_ssim_weight = 0.45
        self.final_boundary_weight = 0.10
        self.final_intensity_weight = 0.20
        self.final_intensity_stats_weight = 0.05

        # Stage-1 deep supervision remains lighter than the final objective,
        # but L1 is increased so coarse restoration also learns amplitude.
        self.coarse_loss_weight = 0.20
        self.coarse_l1_weight = 0.45
        self.coarse_ssim_weight = 0.55

        # Intensity-aware weighting.
        # weight = 1 + bright_boost * target^gamma
        self.intensity_bright_boost = 4.0
        self.intensity_gamma = 1.5
        self.intensity_stats_std_weight = 0.5

        # SSIM settings. The current LOW/HIGH data are normalized float32.
        self.ssim_window_size = 11
        self.ssim_sigma = 1.5
        self.ssim_data_range = 1.0

        # Boundary loss settings.
        self.boundary_eps = 1e-6

        # Gradient clipping improves stability in multi-stage residual training.
        self.grad_clip_norm = 1.0


class SaveConfig:
    def __init__(self) -> None:
        self.run_root = Path("./13_Intensity_Loss_Residual")

        self.checkpoint_every = 10
        self.map_save_every = 5
        self.map_volume_index = 0
        self.map_batch_size = 8
        self.save_initial_map = False

    @property
    def model_root(self) -> Path:
        return self.run_root / "models_enhancement"

    @property
    def latest_model_dir(self) -> Path:
        return self.model_root / "latest"

    @property
    def checkpoint_dir(self) -> Path:
        return self.model_root / "checkpoints"

    @property
    def final_model_dir(self) -> Path:
        return self.model_root / "final"

    @property
    def output_root(self) -> Path:
        return self.run_root / "outputs"

    @property
    def train_map_dir(self) -> Path:
        return self.output_root / "train_maps"

    @property
    def history_dir(self) -> Path:
        return self.output_root / "history"

    @property
    def latest_model_path(self) -> Path:
        return self.latest_model_dir / "model_latest.pth"

    @property
    def final_model_path(self) -> Path:
        return self.final_model_dir / "model_final.pth"

    @property
    def history_csv_path(self) -> Path:
        return self.history_dir / "training_history.csv"

    @property
    def loss_curve_path(self) -> Path:
        return self.history_dir / "training_loss_curve.png"


DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
SAVE = SaveConfig()

X_SIZE, Y_SIZE, Z_SIZE = DATA.high_volume_shape
INPUT_SAMPLE_SHAPE = (DATA.input_channels, Y_SIZE, Z_SIZE)
TARGET_SAMPLE_SHAPE = (DATA.output_channels, Y_SIZE, Z_SIZE)


# ============================================================
# 2. Reproducibility / CUDA
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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass



def create_output_directories() -> None:
    for directory in (
        SAVE.latest_model_dir,
        SAVE.checkpoint_dir,
        SAVE.final_model_dir,
        SAVE.train_map_dir,
        SAVE.history_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. File utilities
# ============================================================


def extract_volume_key(file_path: Path) -> str:
    """
    Train_000_LOW.bin  -> Train_000
    Train_000_HIGH.bin -> Train_000
    """
    return re.sub(
        r"_(LOW|HIGH)$",
        "",
        file_path.stem,
        flags=re.IGNORECASE,
    )



def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]



def get_bin_files(directory: Path) -> list[Path]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.bin"), key=natural_key)

    if not files:
        raise RuntimeError(f"No .bin files found in: {directory}")

    return files



def validate_bin_file(
    file_path: Path,
    shape: tuple[int, ...],
    dtype: Any = DATA.dtype,
    offset: int = DATA.header_size,
) -> None:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dtype = np.dtype(dtype)
    expected_elements = int(np.prod(shape))
    expected_size = offset + expected_elements * dtype.itemsize
    actual_size = file_path.stat().st_size

    if actual_size != expected_size:
        raise ValueError(
            "\nBinary file size mismatch\n"
            f"File            : {file_path}\n"
            f"Actual bytes    : {actual_size:,}\n"
            f"Expected bytes  : {expected_size:,}\n"
            f"Expected shape  : {shape}\n"
            f"Expected dtype  : {dtype}\n"
            f"Header size     : {offset}"
        )



def build_file_pairs(low_dir: Path, high_dir: Path) -> list[dict[str, Any]]:
    low_files = get_bin_files(low_dir)
    high_files = get_bin_files(high_dir)

    low_map = {extract_volume_key(path): path for path in low_files}
    high_map = {extract_volume_key(path): path for path in high_files}

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

        validate_bin_file(low_path, DATA.adjacent_low_shape)
        validate_bin_file(high_path, DATA.high_volume_shape)

        pairs.append(
            {
                "volume_key": key,
                "low_path": low_path,
                "high_path": high_path,
            }
        )

    if not pairs:
        raise RuntimeError("No valid LOW-HIGH training pairs found.")

    return pairs


# ============================================================
# 4. Dataset
# ============================================================


class PAMAdjacentYZDataset(Dataset):
    """
    LOW input is already pre-generated 3-adjacent Y-Z data.

    LOW file:
        [X,C,Y,Z] = [200,3,200,512]

    HIGH file:
        [X,Y,Z] = [200,200,512]

    One sample at x_index:
        input  = LOW[x_index]     -> [3,200,512]
        target = HIGH[x_index]    -> [1,200,512]

    No neighboring-slice construction is performed here.
    """

    def __init__(self, low_dir: Path, high_dir: Path) -> None:
        super().__init__()

        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.pairs = build_file_pairs(self.low_dir, self.high_dir)

        self.num_volumes = len(self.pairs)
        self.samples_per_volume = X_SIZE
        self.total_samples = self.num_volumes * self.samples_per_volume

        self._low_memmaps: dict[int, np.memmap] = {}
        self._high_memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.total_samples

    def _get_low_memmap(self, volume_index: int) -> np.memmap:
        if volume_index not in self._low_memmaps:
            path = self.pairs[volume_index]["low_path"]
            self._low_memmaps[volume_index] = np.memmap(
                filename=path,
                dtype=DATA.dtype,
                mode="r",
                offset=DATA.header_size,
                shape=DATA.adjacent_low_shape,
                order="C",
            )

        return self._low_memmaps[volume_index]

    def _get_high_memmap(self, volume_index: int) -> np.memmap:
        if volume_index not in self._high_memmaps:
            path = self.pairs[volume_index]["high_path"]
            self._high_memmaps[volume_index] = np.memmap(
                filename=path,
                dtype=DATA.dtype,
                mode="r",
                offset=DATA.header_size,
                shape=DATA.high_volume_shape,
                order="C",
            )

        return self._high_memmaps[volume_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Dataset index out of range: {index}. "
                f"Valid range: 0 ~ {len(self) - 1}"
            )

        volume_index, x_index = divmod(index, self.samples_per_volume)
        pair = self.pairs[volume_index]

        low_volume = self._get_low_memmap(volume_index)
        high_volume = self._get_high_memmap(volume_index)

        input_array = np.array(
            low_volume[x_index, :, :, :],
            dtype=np.float32,
            copy=True,
        )

        target_array = np.array(
            high_volume[x_index, :, :],
            dtype=np.float32,
            copy=True,
        )[None, ...]

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
        state = self.__dict__.copy()
        state["_low_memmaps"] = {}
        state["_high_memmaps"] = {}
        return state


class EpochRandomXSliceSampler(Sampler[int]):
    """
    Per epoch:
        1. Shuffle volume order.
        2. Sample N unique X positions per volume.
        3. Sort X positions within each volume for better I/O locality.
    """

    def __init__(
        self,
        dataset: PAMAdjacentYZDataset,
        slices_per_volume: int,
        seed: int = 42,
    ) -> None:
        if slices_per_volume <= 0:
            raise ValueError("slices_per_volume must be > 0")

        if slices_per_volume > dataset.samples_per_volume:
            raise ValueError(
                f"slices_per_volume={slices_per_volume} exceeds available "
                f"X positions={dataset.samples_per_volume}"
            )

        self.dataset = dataset
        self.slices_per_volume = int(slices_per_volume)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.dataset.num_volumes * self.slices_per_volume

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        volume_order = rng.permutation(self.dataset.num_volumes)

        global_indices: list[int] = []

        for volume_index in volume_order:
            x_indices = rng.choice(
                self.dataset.samples_per_volume,
                size=self.slices_per_volume,
                replace=False,
            )
            x_indices.sort()

            base_index = int(volume_index) * self.dataset.samples_per_volume
            global_indices.extend(
                base_index + int(x_index)
                for x_index in x_indices
            )

        return iter(global_indices)


# ============================================================
# 5. Model utility layers
# ============================================================


def _group_count(channels: int, maximum_groups: int = 8) -> int:
    groups = min(maximum_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class LearnableResidualScale(nn.Module):
    """
    Positive bounded scalar in (0, max_scale).

    This avoids letting a residual branch dominate the center LOW image at
    initialization. The requested initial scale is represented by a learnable
    logit and optimized end-to-end.
    """

    def __init__(self, init_value: float = 0.1, max_scale: float = 1.0) -> None:
        super().__init__()

        if not 0.0 < init_value < max_scale:
            raise ValueError(
                f"Require 0 < init_value < max_scale, got "
                f"{init_value} and {max_scale}"
            )

        ratio = init_value / max_scale
        logit = math.log(ratio / (1.0 - ratio))

        self.logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        self.max_scale = float(max_scale)

    def forward(self) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.logit)


class ChannelAttention(nn.Module):
    """Lightweight squeeze-excitation channel modulation."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.body(x)


class RestorationBlock(nn.Module):
    """
    Efficient residual restoration block.

    Pointwise expansion -> depthwise spatial filtering -> GELU ->
    channel attention -> pointwise projection -> residual addition.

    A learnable per-channel layer scale starts at 0.1 to keep optimization
    conservative and stable.
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        hidden = channels * expansion
        padding = dilation

        self.norm = nn.GroupNorm(
            _group_count(channels),
            channels,
        )

        self.expand = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            bias=True,
        )

        self.depthwise = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            groups=hidden,
            bias=True,
        )

        self.activation = nn.GELU()
        self.attention = ChannelAttention(hidden)

        self.project = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.attention(x)
        x = self.project(x)

        return residual + self.layer_scale * x


class BlockStack(nn.Sequential):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        dilations: tuple[int, ...] | None = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")

        if dilations is None:
            dilations = tuple(1 for _ in range(num_blocks))

        if len(dilations) != num_blocks:
            raise ValueError(
                f"Expected {num_blocks} dilations, got {dilations}"
            )

        super().__init__(
            *[
                RestorationBlock(channels, dilation=dilation)
                for dilation in dilations
            ]
        )


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PixelShuffleUp(nn.Module):
    """Learned 2x upsampling for sharper restoration than plain interpolation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.expand = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.shuffle = nn.PixelShuffle(2)
        self.refine = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.shuffle(self.expand(x)))


class DecoderStage(nn.Module):
    """
    PixelShuffle upsampling + direct skip concatenation + compact restoration.

    No AFF/SFA/dense hierarchy is used. The skip path is intentionally simple
    to reduce excessive feature averaging.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_blocks: int,
    ) -> None:
        super().__init__()

        self.up = PixelShuffleUp(in_channels, out_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
        )

        self.blocks = BlockStack(out_channels, num_blocks)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        return x


# ============================================================
# 6. Fixed edge and wavelet guidance
# ============================================================


class FixedEdgeExtractor(nn.Module):
    """
    Compute fixed full-resolution edge priors from the center LOW plane.

    Returns:
        gradient_magnitude : [B,1,H,W]
        laplacian          : [B,1,H,W] (signed)
        laplacian_abs      : [B,1,H,W]
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
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

        laplacian = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer("sobel_y", sobel_y, persistent=True)
        self.register_buffer("laplacian", laplacian, persistent=True)

    def forward(
        self,
        center: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if center.ndim != 4 or center.size(1) != 1:
            raise ValueError(
                "FixedEdgeExtractor expects [B,1,H,W], "
                f"got {tuple(center.shape)}"
            )

        original_dtype = center.dtype
        x = center.float()
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")

        gx = F.conv2d(x, self.sobel_x)
        gy = F.conv2d(x, self.sobel_y)
        lap = F.conv2d(x, self.laplacian)

        grad = torch.sqrt(gx.square() + gy.square() + self.eps)
        lap_abs = lap.abs()

        return (
            grad.to(dtype=original_dtype),
            lap.to(dtype=original_dtype),
            lap_abs.to(dtype=original_dtype),
        )


class HaarDWT(nn.Module):
    """
    Parameter-free orthonormal-style 2D Haar decomposition by tensor slicing.

    Input:
        [B,C,H,W], where H and W are even.

    Returns:
        LL, LH, HL, HH at [B,C,H/2,W/2].
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise ValueError(
                "HaarDWT requires even H and W, got "
                f"{tuple(x.shape[-2:])}"
            )

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        # Scale 0.5 preserves orthonormal Haar energy convention.
        ll = 0.5 * (a + b + c + d)
        lh = 0.5 * (-a - b + c + d)
        hl = 0.5 * (-a + b - c + d)
        hh = 0.5 * (a - b - c + d)

        return ll, lh, hl, hh


# ============================================================
# 7. architecture branches
# ============================================================


class MainRestorationBranch(nn.Module):
    """
    Compact multi-scale encoder-decoder for global structure and continuity.

    Input:
        [B,3,H,W]

    Output:
        full-resolution feature [B,C,H,W]
        main residual          [B,1,H,W]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        c1 = config.base_channels
        c2 = min(c1 * 2, config.max_channels)
        c3 = min(c1 * 4, config.max_channels)
        c4 = min(c1 * 8, config.max_channels)

        self.stem = nn.Conv2d(
            config.in_channels,
            c1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.encoder1 = BlockStack(c1, config.encoder_blocks[0])
        self.down1 = Downsample(c1, c2)

        self.encoder2 = BlockStack(c2, config.encoder_blocks[1])
        self.down2 = Downsample(c2, c3)

        self.encoder3 = BlockStack(c3, config.encoder_blocks[2])
        self.down3 = Downsample(c3, c4)

        bottleneck_dilations = tuple(
            (1, 2, 3, 1)[index % 4]
            for index in range(config.bottleneck_blocks)
        )
        self.bottleneck = BlockStack(
            c4,
            config.bottleneck_blocks,
            dilations=bottleneck_dilations,
        )

        self.decoder3 = DecoderStage(
            c4,
            c3,
            c3,
            config.decoder_blocks[0],
        )
        self.decoder2 = DecoderStage(
            c3,
            c2,
            c2,
            config.decoder_blocks[1],
        )
        self.decoder1 = DecoderStage(
            c2,
            c1,
            c1,
            config.decoder_blocks[2],
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(c1, config.out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.encoder1(self.stem(x))
        e2 = self.encoder2(self.down1(e1))
        e3 = self.encoder3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))

        d3 = self.decoder3(b, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        residual = self.residual_head(d1)
        return d1, residual


class EdgeDetailBranch(nn.Module):
    """
    Full-resolution edge branch.

    Inputs:
        center LOW
        Sobel magnitude
        signed Laplacian
        absolute Laplacian

    Output:
        [B, edge_channels, H, W]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        c = config.edge_channels
        self.stem = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.blocks = BlockStack(c, 3, dilations=(1, 2, 1))

    def forward(
        self,
        center: torch.Tensor,
        grad: torch.Tensor,
        lap: torch.Tensor,
        lap_abs: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([center, grad, lap, lap_abs], dim=1)
        x = self.stem(x)
        return self.blocks(x)


class WaveletFrequencyBranch(nn.Module):
    """
    Haar high-frequency branch specialized for thin vessels and boundaries.

    Input bands:
        LH, HL, HH, HF energy

    Processing occurs at H/2 x W/2 and is returned to full resolution using
    PixelShuffle rather than simple bilinear interpolation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        c = config.wavelet_channels

        self.stem = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )
        self.blocks = BlockStack(c, 3, dilations=(1, 2, 1))
        self.up = PixelShuffleUp(c, c)

    def forward(
        self,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
        hf_energy: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([lh, hl, hh, hf_energy], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.up(x)


class GatedDetailFusion(nn.Module):
    """
    Fuse edge and wavelet detail features and inject them conservatively.

    Gate input:
        main full-resolution feature
        fused detail feature
        center LOW
        Sobel magnitude

    Gate output:
        one spatial gate [B,1,H,W]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        c_main = config.base_channels
        c_edge = config.edge_channels
        c_wavelet = config.wavelet_channels
        c_detail = config.detail_channels

        self.detail_fuse = nn.Sequential(
            nn.Conv2d(
                c_edge + c_wavelet,
                c_detail,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            RestorationBlock(c_detail),
        )

        self.detail_head = nn.Sequential(
            nn.Conv2d(c_detail, c_detail, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(c_detail, 1, kernel_size=3, padding=1, bias=True),
        )

        self.gate = nn.Conv2d(
            c_main + c_detail + 2,
            1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        nn.init.constant_(self.gate.bias, config.detail_gate_bias_init)

    def forward(
        self,
        main_feature: torch.Tensor,
        edge_feature: torch.Tensor,
        wavelet_feature: torch.Tensor,
        center: torch.Tensor,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_feature.shape[-2:] != main_feature.shape[-2:]:
            edge_feature = F.interpolate(
                edge_feature,
                size=main_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if wavelet_feature.shape[-2:] != main_feature.shape[-2:]:
            wavelet_feature = F.interpolate(
                wavelet_feature,
                size=main_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        detail_feature = self.detail_fuse(
            torch.cat([edge_feature, wavelet_feature], dim=1)
        )

        detail_residual = self.detail_head(detail_feature)

        gate_input = torch.cat(
            [main_feature, detail_feature, center, grad],
            dim=1,
        )
        detail_gate = torch.sigmoid(self.gate(gate_input))

        gated_detail_residual = detail_gate * detail_residual

        return gated_detail_residual, detail_gate, detail_feature


class FullResolutionDetailRefiner(nn.Module):
    """
    Stage-2 full-resolution refiner.

    It receives explicit evidence from:
        coarse prediction
        original center LOW
        coarse - center residual
        Sobel magnitude
        full-resolution wavelet HF energy

    No spatial downsampling is used in Stage 2, preventing another loss of thin
    vessels. Dilated residual restoration blocks expand the receptive field.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        c = config.refiner_channels
        self.stem = nn.Sequential(
            nn.Conv2d(5, c, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.blocks = BlockStack(
            c,
            len(config.refiner_dilations),
            dilations=config.refiner_dilations,
        )

        self.residual_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(c, 1, kernel_size=3, padding=1, bias=True),
        )

        self.gate = nn.Conv2d(
            c + 2,
            1,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.constant_(self.gate.bias, config.refine_gate_bias_init)

    def forward(
        self,
        coarse: torch.Tensor,
        center: torch.Tensor,
        grad: torch.Tensor,
        hf_energy_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_delta = coarse - center

        x = torch.cat(
            [coarse, center, coarse_delta, grad, hf_energy_full],
            dim=1,
        )

        feature = self.blocks(self.stem(x))
        residual = self.residual_head(feature)

        gate_input = torch.cat([feature, grad, hf_energy_full], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))

        return gate * residual, gate, feature


# ============================================================
# 8. Full model
# ============================================================


class ResidualEdgeWaveletMultiStageNet(nn.Module):
    """
    : Residual Edge-Wavelet Multi-Stage PAM Enhancement Network.

    Input:
        [B,3,H,W]
        channel 0 = LOW[x-1]
        channel 1 = LOW[x]      (center plane)
        channel 2 = LOW[x+1]

    Output by default:
        final prediction [B,1,H,W]

    Training can request intermediate outputs:
        model(x, return_intermediates=True)

    The default forward output is a single tensor for compatibility with the
    existing evaluator.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
        max_channels: int = 128,
        use_edge_branch: bool = True,
        use_wavelet_branch: bool = True,
        use_stage2: bool = True,
    ) -> None:
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                "currently expects exactly 3 adjacent input channels. "
                f"Got {in_channels}."
            )

        if out_channels != 1:
            raise ValueError(
                "currently expects one output channel. "
                f"Got {out_channels}."
            )

        config = ModelConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            max_channels=max_channels,
            use_edge_branch=use_edge_branch,
            use_wavelet_branch=use_wavelet_branch,
            use_stage2=use_stage2,
        )
        self.config = config

        self.edge_extractor = FixedEdgeExtractor()
        self.haar_dwt = HaarDWT()

        self.main_branch = MainRestorationBranch(config)

        self.edge_branch = EdgeDetailBranch(config)
        self.wavelet_branch = WaveletFrequencyBranch(config)
        self.detail_fusion = GatedDetailFusion(config)

        self.refiner = FullResolutionDetailRefiner(config)

        self.alpha = LearnableResidualScale(config.alpha_init)
        self.beta = LearnableResidualScale(config.beta_init)
        self.gamma = LearnableResidualScale(config.gamma_init)

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor,
        multiple: int = 8,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        height, width = x.shape[-2:]

        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = (pad_left, pad_right, pad_top, pad_bottom)

        if any(padding):
            x = F.pad(x, padding, mode="reflect")

        return x, padding

    @staticmethod
    def _remove_padding(
        x: torch.Tensor,
        padding: tuple[int, int, int, int],
    ) -> torch.Tensor:
        pad_left, pad_right, pad_top, pad_bottom = padding

        h_end = x.shape[-2] - pad_bottom if pad_bottom > 0 else x.shape[-2]
        w_end = x.shape[-1] - pad_right if pad_right > 0 else x.shape[-1]

        return x[..., pad_top:h_end, pad_left:w_end]

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input [B,3,H,W], got {tuple(x.shape)}"
            )

        if x.size(1) != 3:
            raise ValueError(
                f"Expected 3 input channels, got {x.size(1)}"
            )

        original_height, original_width = x.shape[-2:]

        # Pad to a multiple of 8 for three encoder downsampling operations and
        # even Haar decomposition. Current 200x512 becomes 200x512 already for
        # /8 in height? 200 is divisible by 8, so no padding is needed.
        x_pad, padding = self._pad_to_multiple(x, multiple=8)

        center = x_pad[:, 1:2, :, :]

        # ----------------------------------------------------
        # Fixed structural guidance
        # ----------------------------------------------------
        grad, lap, lap_abs = self.edge_extractor(center)
        _, lh, hl, hh = self.haar_dwt(center)
        hf_energy = torch.sqrt(
            lh.square() + hl.square() + hh.square() + 1e-6
        )
        hf_energy_full = F.interpolate(
            hf_energy,
            size=center.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Stage 1: main restoration path
        # ----------------------------------------------------
        main_feature, main_residual = self.main_branch(x_pad)

        # ----------------------------------------------------
        # Stage 1: edge / wavelet high-frequency detail path
        # ----------------------------------------------------
        if self.config.use_edge_branch:
            edge_feature = self.edge_branch(center, grad, lap, lap_abs)
        else:
            edge_feature = torch.zeros(
                center.shape[0],
                self.config.edge_channels,
                center.shape[-2],
                center.shape[-1],
                dtype=center.dtype,
                device=center.device,
            )

        if self.config.use_wavelet_branch:
            wavelet_feature = self.wavelet_branch(lh, hl, hh, hf_energy)
        else:
            wavelet_feature = torch.zeros(
                center.shape[0],
                self.config.wavelet_channels,
                center.shape[-2],
                center.shape[-1],
                dtype=center.dtype,
                device=center.device,
            )

        gated_detail_residual, detail_gate, detail_feature = self.detail_fusion(
            main_feature=main_feature,
            edge_feature=edge_feature,
            wavelet_feature=wavelet_feature,
            center=center,
            grad=grad,
        )

        alpha = self.alpha()
        beta = self.beta()

        coarse = (
            center
            + alpha * main_residual
            + beta * gated_detail_residual
        )

        # ----------------------------------------------------
        # Stage 2: full-resolution detail refinement
        # ----------------------------------------------------
        if self.config.use_stage2:
            refine_residual, refine_gate, refine_feature = self.refiner(
                coarse=coarse,
                center=center,
                grad=grad,
                hf_energy_full=hf_energy_full,
            )

            gamma = self.gamma()
            final = coarse + gamma * refine_residual
        else:
            refine_residual = torch.zeros_like(coarse)
            refine_gate = torch.zeros_like(coarse)
            refine_feature = torch.empty(
                0,
                dtype=coarse.dtype,
                device=coarse.device,
            )
            gamma = torch.zeros((), dtype=coarse.dtype, device=coarse.device)
            final = coarse

        # Remove dynamic padding.
        center = self._remove_padding(center, padding)
        main_residual = self._remove_padding(main_residual, padding)
        gated_detail_residual = self._remove_padding(
            gated_detail_residual,
            padding,
        )
        detail_gate = self._remove_padding(detail_gate, padding)
        grad = self._remove_padding(grad, padding)
        hf_energy_full = self._remove_padding(hf_energy_full, padding)
        coarse = self._remove_padding(coarse, padding)
        refine_residual = self._remove_padding(refine_residual, padding)
        refine_gate = self._remove_padding(refine_gate, padding)
        final = self._remove_padding(final, padding)

        expected_shape = (original_height, original_width)
        if final.shape[-2:] != expected_shape:
            raise RuntimeError(
                f"output spatial shape mismatch: "
                f"{tuple(final.shape[-2:])} != {expected_shape}"
            )

        if not return_intermediates:
            return final

        return {
            "prediction": final,
            "final": final,
            "coarse": coarse,
            "center": center,
            "main_residual": main_residual,
            "detail_residual": gated_detail_residual,
            "refine_residual": refine_residual,
            "detail_gate": detail_gate,
            "refine_gate": refine_gate,
            "edge_magnitude": grad,
            "hf_energy": hf_energy_full,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }


# Convenient aliases for dynamic evaluators.
PAMEnhancer = ResidualEdgeWaveletMultiStageNet
UNet = ResidualEdgeWaveletMultiStageNet
Model = ResidualEdgeWaveletMultiStageNet


# ============================================================
# 9. Loss
# ============================================================


class SSIMLoss(nn.Module):
    """
    Differentiable single-channel SSIM loss.

    loss = 1 - mean(SSIM)

    The current PAM training data are normalized float32, so data_range=1.0 is
    used by default. No prediction clamping is applied, preserving gradients
    even when the linear model output temporarily leaves [0, 1].
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if data_range <= 0:
            raise ValueError("data_range must be > 0")

        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        self.eps = float(eps)

        coords = torch.arange(self.window_size, dtype=torch.float32)
        coords = coords - (self.window_size - 1) / 2.0
        gaussian_1d = torch.exp(-(coords.square()) / (2.0 * self.sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        window_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        self.register_buffer(
            "window",
            window_2d.reshape(1, 1, self.window_size, self.window_size),
            persistent=True,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "SSIMLoss shape mismatch: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )

        if prediction.ndim != 4:
            raise ValueError(
                f"SSIMLoss expects [B,C,H,W], got {tuple(prediction.shape)}"
            )

        channels = prediction.size(1)
        window = self.window.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).expand(channels, 1, -1, -1)

        padding = self.window_size // 2

        mu_x = F.conv2d(
            prediction,
            window,
            padding=padding,
            groups=channels,
        )
        mu_y = F.conv2d(
            target,
            window,
            padding=padding,
            groups=channels,
        )

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(
            prediction.square(),
            window,
            padding=padding,
            groups=channels,
        ) - mu_x_sq

        sigma_y_sq = F.conv2d(
            target.square(),
            window,
            padding=padding,
            groups=channels,
        ) - mu_y_sq

        sigma_xy = F.conv2d(
            prediction * target,
            window,
            padding=padding,
            groups=channels,
        ) - mu_xy

        # Small numerical negatives may appear from floating-point subtraction.
        sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
        sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (
            (mu_x_sq + mu_y_sq + c1)
            * (sigma_x_sq + sigma_y_sq + c2)
        )

        ssim_map = numerator / (denominator + self.eps)
        ssim_value = ssim_map.mean()

        return 1.0 - ssim_value


class BoundaryGradientLoss(nn.Module):
    """
    Sobel boundary loss for vessel sharpness and edge alignment.

    The loss compares signed horizontal/vertical gradients and gradient
    magnitude. Comparing signed gradients preserves edge orientation, while the
    magnitude term directly penalizes blurred or weakened vessel boundaries.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3) / 8.0

        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer("sobel_y", sobel_y, persistent=True)

    def _gradients(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels = x.size(1)
        kernel_x = self.sobel_x.to(device=x.device, dtype=x.dtype).expand(
            channels, 1, -1, -1
        )
        kernel_y = self.sobel_y.to(device=x.device, dtype=x.dtype).expand(
            channels, 1, -1, -1
        )

        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx = F.conv2d(x_pad, kernel_x, groups=channels)
        gy = F.conv2d(x_pad, kernel_y, groups=channels)
        magnitude = torch.sqrt(gx.square() + gy.square() + self.eps)
        return gx, gy, magnitude

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "BoundaryGradientLoss shape mismatch: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )

        pred_gx, pred_gy, pred_mag = self._gradients(prediction)
        target_gx, target_gy, target_mag = self._gradients(target)

        directional_loss = 0.5 * (
            F.l1_loss(pred_gx, target_gx)
            + F.l1_loss(pred_gy, target_gy)
        )
        magnitude_loss = F.l1_loss(pred_mag, target_mag)

        return 0.5 * directional_loss + 0.5 * magnitude_loss


class TargetWeightedIntensityL1Loss(nn.Module):
    """
    Target-intensity-aware weighted L1 loss.

    Bright pixels in HIGH receive larger weights, while background pixels keep
    a base weight of 1.0:

        weight = 1 + bright_boost * target^gamma

    This directly increases the gradient contribution of strong PAM signals
    and bright vessel regions without discarding the background.
    """

    def __init__(
        self,
        bright_boost: float = 4.0,
        gamma: float = 1.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if bright_boost < 0:
            raise ValueError("bright_boost must be >= 0")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")

        self.bright_boost = float(bright_boost)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "TargetWeightedIntensityL1Loss shape mismatch: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )

        # The current training data are expected to be normalized to [0, 1].
        # detach() prevents gradients from flowing through the weighting map.
        target_importance = (
            target.detach()
            .clamp(min=0.0, max=1.0)
            .pow(self.gamma)
        )

        weights = 1.0 + self.bright_boost * target_importance
        absolute_error = torch.abs(prediction - target)

        weighted_error = (weights * absolute_error).sum()
        normalization = weights.sum().clamp_min(self.eps)

        return weighted_error / normalization


class IntensityStatisticsLoss(nn.Module):
    """
    Match target-weighted intensity mean and standard deviation.

    This auxiliary term is intentionally small. Its role is to discourage:
        - globally dark predictions,
        - compressed dynamic range,
        - weakened bright-vessel contrast.

    Statistics are computed independently per sample and then averaged.
    """

    def __init__(
        self,
        bright_boost: float = 4.0,
        gamma: float = 1.5,
        std_weight: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if bright_boost < 0:
            raise ValueError("bright_boost must be >= 0")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if std_weight < 0:
            raise ValueError("std_weight must be >= 0")

        self.bright_boost = float(bright_boost)
        self.gamma = float(gamma)
        self.std_weight = float(std_weight)
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "IntensityStatisticsLoss shape mismatch: "
                f"prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )

        target_importance = (
            target.detach()
            .clamp(min=0.0, max=1.0)
            .pow(self.gamma)
        )
        weights = 1.0 + self.bright_boost * target_importance

        dims = (1, 2, 3)
        denominator = weights.sum(
            dim=dims,
            keepdim=True,
        ).clamp_min(self.eps)

        pred_mean = (weights * prediction).sum(
            dim=dims,
            keepdim=True,
        ) / denominator

        target_mean = (weights * target).sum(
            dim=dims,
            keepdim=True,
        ) / denominator

        pred_variance = (
            weights * (prediction - pred_mean).square()
        ).sum(
            dim=dims,
            keepdim=True,
        ) / denominator

        target_variance = (
            weights * (target - target_mean).square()
        ).sum(
            dim=dims,
            keepdim=True,
        ) / denominator

        pred_std = torch.sqrt(pred_variance + self.eps)
        target_std = torch.sqrt(target_variance + self.eps)

        mean_loss = torch.abs(pred_mean - target_mean).mean()
        std_loss = torch.abs(pred_std - target_std).mean()

        return mean_loss + self.std_weight * std_loss


class MultiStageStructureLoss(nn.Module):
    """
    .1 intensity-aware structure-preserving PAM loss.

    Final-stage objective:
        0.20 * L1
      + 0.45 * SSIM loss
      + 0.10 * boundary gradient loss
      + 0.20 * target-weighted intensity L1
      + 0.05 * intensity-statistics loss

    Stage-1 coarse deep supervision:
        0.20 * (0.45 * coarse L1 + 0.55 * coarse SSIM loss)

    Total:
        final_composite + coarse_weight * coarse_composite

    SSIM remains the largest single final-stage term, while explicit intensity
    supervision prevents the network from optimizing structure at the expense
    of PAM amplitude.
    """

    def __init__(
        self,
        final_l1_weight: float = 0.20,
        final_ssim_weight: float = 0.45,
        final_boundary_weight: float = 0.10,
        final_intensity_weight: float = 0.20,
        final_intensity_stats_weight: float = 0.05,
        coarse_weight: float = 0.20,
        coarse_l1_weight: float = 0.45,
        coarse_ssim_weight: float = 0.55,
        ssim_window_size: int = 11,
        ssim_sigma: float = 1.5,
        ssim_data_range: float = 1.0,
        boundary_eps: float = 1e-6,
        intensity_bright_boost: float = 4.0,
        intensity_gamma: float = 1.5,
        intensity_stats_std_weight: float = 0.5,
    ) -> None:
        super().__init__()

        final_sum = (
            final_l1_weight
            + final_ssim_weight
            + final_boundary_weight
            + final_intensity_weight
            + final_intensity_stats_weight
        )
        coarse_sum = coarse_l1_weight + coarse_ssim_weight

        if not math.isclose(final_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Final loss weights must sum to 1.0, "
                f"got {final_sum:.6f}"
            )
        if not math.isclose(coarse_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Coarse loss weights must sum to 1.0, "
                f"got {coarse_sum:.6f}"
            )
        if coarse_weight < 0:
            raise ValueError("coarse_weight must be >= 0")

        self.final_l1_weight = float(final_l1_weight)
        self.final_ssim_weight = float(final_ssim_weight)
        self.final_boundary_weight = float(final_boundary_weight)
        self.final_intensity_weight = float(final_intensity_weight)
        self.final_intensity_stats_weight = float(
            final_intensity_stats_weight
        )

        self.coarse_weight = float(coarse_weight)
        self.coarse_l1_weight = float(coarse_l1_weight)
        self.coarse_ssim_weight = float(coarse_ssim_weight)

        self.ssim_loss = SSIMLoss(
            window_size=ssim_window_size,
            sigma=ssim_sigma,
            data_range=ssim_data_range,
        )
        self.boundary_loss = BoundaryGradientLoss(eps=boundary_eps)
        self.intensity_loss = TargetWeightedIntensityL1Loss(
            bright_boost=intensity_bright_boost,
            gamma=intensity_gamma,
        )
        self.intensity_stats_loss = IntensityStatisticsLoss(
            bright_boost=intensity_bright_boost,
            gamma=intensity_gamma,
            std_weight=intensity_stats_std_weight,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        final = outputs["final"]
        coarse = outputs["coarse"]

        # Final-stage losses.
        final_l1 = F.l1_loss(final, target)
        final_ssim = self.ssim_loss(final, target)
        final_boundary = self.boundary_loss(final, target)
        final_intensity = self.intensity_loss(final, target)
        final_intensity_stats = self.intensity_stats_loss(final, target)

        # Stage-1 coarse deep supervision.
        coarse_l1 = F.l1_loss(coarse, target)
        coarse_ssim = self.ssim_loss(coarse, target)

        final_composite = (
            self.final_l1_weight * final_l1
            + self.final_ssim_weight * final_ssim
            + self.final_boundary_weight * final_boundary
            + self.final_intensity_weight * final_intensity
            + self.final_intensity_stats_weight * final_intensity_stats
        )

        coarse_composite = (
            self.coarse_l1_weight * coarse_l1
            + self.coarse_ssim_weight * coarse_ssim
        )

        total = final_composite + self.coarse_weight * coarse_composite

        # SSIM score is easier to interpret in logs than SSIM loss.
        final_ssim_score = 1.0 - final_ssim
        coarse_ssim_score = 1.0 - coarse_ssim

        return total, {
            "final_l1": final_l1.detach(),
            "final_ssim_loss": final_ssim.detach(),
            "final_ssim": final_ssim_score.detach(),
            "final_boundary": final_boundary.detach(),
            "final_intensity": final_intensity.detach(),
            "final_intensity_stats": final_intensity_stats.detach(),
            "coarse_l1": coarse_l1.detach(),
            "coarse_ssim_loss": coarse_ssim.detach(),
            "coarse_ssim": coarse_ssim_score.detach(),
            "final_composite": final_composite.detach(),
            "coarse_composite": coarse_composite.detach(),
        }


# ============================================================
# 10. Training history
# ============================================================


class HistoryRow:
    def __init__(
        self,
        epoch: int,
        total_loss: float,
        final_l1: float,
        final_ssim: float,
        final_boundary: float,
        final_intensity: float,
        final_intensity_stats: float,
        coarse_l1: float,
        coarse_ssim: float,
        alpha: float,
        beta: float,
        gamma: float,
        learning_rate: float,
        epoch_seconds: float,
    ) -> None:
        self.epoch = int(epoch)
        self.total_loss = float(total_loss)
        self.final_l1 = float(final_l1)
        self.final_ssim = float(final_ssim)
        self.final_boundary = float(final_boundary)
        self.final_intensity = float(final_intensity)
        self.final_intensity_stats = float(final_intensity_stats)
        self.coarse_l1 = float(coarse_l1)
        self.coarse_ssim = float(coarse_ssim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.epoch_seconds = float(epoch_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "final_l1": self.final_l1,
            "final_ssim": self.final_ssim,
            "final_boundary": self.final_boundary,
            "final_intensity": self.final_intensity,
            "final_intensity_stats": self.final_intensity_stats,
            "coarse_l1": self.coarse_l1,
            "coarse_ssim": self.coarse_ssim,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "epoch_seconds": self.epoch_seconds,
        }


# ============================================================
# 11. MAP visualization
# ============================================================



def load_full_training_volume(
    dataset: PAMAdjacentYZDataset,
    volume_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if volume_index < 0 or volume_index >= dataset.num_volumes:
        raise IndexError(
            f"volume_index={volume_index} outside "
            f"0~{dataset.num_volumes - 1}"
        )

    pair = dataset.pairs[volume_index]

    low_volume = np.memmap(
        filename=pair["low_path"],
        dtype=DATA.dtype,
        mode="r",
        offset=DATA.header_size,
        shape=DATA.adjacent_low_shape,
        order="C",
    )

    high_volume = np.memmap(
        filename=pair["high_path"],
        dtype=DATA.dtype,
        mode="r",
        offset=DATA.header_size,
        shape=DATA.high_volume_shape,
        order="C",
    )

    low_copy = np.asarray(low_volume, dtype=np.float32).copy()
    high_copy = np.asarray(high_volume, dtype=np.float32).copy()

    del low_volume
    del high_volume

    return low_copy, high_copy



def robust_limits(
    images: list[np.ndarray],
    upper_percentile: float = 99.8,
) -> tuple[float, float]:
    values = np.concatenate([image.ravel() for image in images])
    v_min = float(np.percentile(values, 0.0))
    v_max = float(np.percentile(values, upper_percentile))

    if v_max <= v_min:
        v_max = v_min + 1e-8

    return v_min, v_max



def save_training_map(
    model: nn.Module,
    dataset: PAMAdjacentYZDataset,
    device: torch.device,
    epoch: int,
) -> None:
    model_was_training = model.training
    model.eval()

    low_volume, high_volume = load_full_training_volume(
        dataset,
        SAVE.map_volume_index,
    )

    # Central channel of pre-generated 3-adjacent LOW volume.
    low_center = low_volume[:, 1, :, :]

    prediction_volume = np.empty(
        DATA.high_volume_shape,
        dtype=np.float32,
    )

    amp_enabled = TRAIN.use_amp and device.type == "cuda"

    with torch.inference_mode():
        for start in range(0, X_SIZE, SAVE.map_batch_size):
            end = min(start + SAVE.map_batch_size, X_SIZE)

            batch = torch.from_numpy(
                np.array(low_volume[start:end], dtype=np.float32, copy=True)
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if TRAIN.use_channels_last and device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(batch)

            prediction_volume[start:end] = (
                prediction[:, 0].float().cpu().numpy()
            )

    low_map = np.max(low_center, axis=0)
    pred_map = np.max(prediction_volume, axis=0)
    high_map = np.max(high_volume, axis=0)
    error_map = np.abs(high_map - pred_map)

    # Rotate 90 degrees clockwise to match the current evaluator convention.
    low_map = np.rot90(low_map, k=-1)
    pred_map = np.rot90(pred_map, k=-1)
    high_map = np.rot90(high_map, k=-1)
    error_map = np.rot90(error_map, k=-1)

    v_min, v_max = robust_limits([low_map, pred_map, high_map])
    error_max = float(np.percentile(error_map, 99.8))
    if error_max <= 0:
        error_max = 1e-8

    fig, axes = plt.subplots(1, 4, figsize=(22, 10))

    panels = [
        (low_map, "LOW Y-Z MAP", "hot", v_min, v_max),
        (pred_map, "Prediction Y-Z MAP", "hot", v_min, v_max),
        (high_map, "HIGH Y-Z MAP", "hot", v_min, v_max),
        (error_map, "Absolute Error Y-Z MAP", "viridis", 0.0, error_max),
    ]

    for axis, (image, title, cmap, p_min, p_max) in zip(axes, panels):
        rendered = axis.imshow(
            image,
            cmap=cmap,
            origin="upper",
            aspect="auto",
            vmin=p_min,
            vmax=p_max,
        )
        axis.set_title(title)
        axis.set_xlabel("Y")
        axis.set_ylabel("Z")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        f".1 PAM Intensity-Aware Enhancement | Epoch {epoch:03d} | 90° clockwise"
    )
    fig.tight_layout()

    save_path = SAVE.train_map_dir / f"epoch_{epoch:03d}_map.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    del low_volume
    del high_volume
    del prediction_volume

    if model_was_training:
        model.train()


# ============================================================
# 12. Checkpoint / history utilities
# ============================================================



def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable



def model_metadata(samples_per_epoch: int) -> dict[str, Any]:
    return {
        "architecture": "ResidualEdgeWaveletMultiStageNet",
        "input_shape": list(INPUT_SAMPLE_SHAPE),
        "target_shape": list(TARGET_SAMPLE_SHAPE),
        "data_low_shape": list(DATA.adjacent_low_shape),
        "data_high_shape": list(DATA.high_volume_shape),
        "dtype": str(np.dtype(DATA.dtype)),
        "header_size": DATA.header_size,
        "center_channel": 1,
        "main_branch": "residual encoder-decoder with PixelShuffle decoder",
        "edge_branch": MODEL.use_edge_branch,
        "edge_guidance": "fixed Sobel magnitude + signed/absolute Laplacian",
        "wavelet_branch": MODEL.use_wavelet_branch,
        "wavelet_guidance": "Haar LH/HL/HH + HF energy",
        "detail_fusion": "spatial sigmoid gate with negative-bias initialization",
        "stage2_refinement": MODEL.use_stage2,
        "residual_formula": (
            "coarse=center+alpha*main_residual+beta*detail_gate*detail_residual; "
            "final=coarse+gamma*refine_gate*refine_residual"
        ),
        "base_channels": MODEL.base_channels,
        "max_channels": MODEL.max_channels,
        "alpha_init": MODEL.alpha_init,
        "beta_init": MODEL.beta_init,
        "gamma_init": MODEL.gamma_init,
        "loss": (
            "0.20*final_L1 + 0.45*final_SSIM_loss + "
            "0.10*final_boundary + 0.20*final_intensity + "
            "0.05*final_intensity_stats + "
            "0.20*(0.45*coarse_L1 + 0.55*coarse_SSIM_loss)"
        ),
        "final_l1_weight": TRAIN.final_l1_weight,
        "final_ssim_weight": TRAIN.final_ssim_weight,
        "final_boundary_weight": TRAIN.final_boundary_weight,
        "final_intensity_weight": TRAIN.final_intensity_weight,
        "final_intensity_stats_weight": TRAIN.final_intensity_stats_weight,
        "intensity_bright_boost": TRAIN.intensity_bright_boost,
        "intensity_gamma": TRAIN.intensity_gamma,
        "intensity_stats_std_weight": TRAIN.intensity_stats_std_weight,
        "coarse_loss_weight": TRAIN.coarse_loss_weight,
        "coarse_l1_weight": TRAIN.coarse_l1_weight,
        "coarse_ssim_weight": TRAIN.coarse_ssim_weight,
        "ssim_window_size": TRAIN.ssim_window_size,
        "ssim_sigma": TRAIN.ssim_sigma,
        "ssim_data_range": TRAIN.ssim_data_range,
        "optimizer": "AdamW",
        "learning_rate": TRAIN.learning_rate,
        "weight_decay": TRAIN.weight_decay,
        "physical_batch_size": TRAIN.batch_size,
        "gradient_accumulation_steps": TRAIN.gradient_accumulation_steps,
        "effective_batch_size": (
            TRAIN.batch_size * TRAIN.gradient_accumulation_steps
        ),
        "slices_per_volume_per_epoch": TRAIN.slices_per_volume_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "use_amp": TRAIN.use_amp,
        "use_channels_last": TRAIN.use_channels_last,
        "final_output_activation": "None (Linear)",
        "clamping": "None",
    }



def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    history_row: HistoryRow,
    samples_per_epoch: int,
    path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history_row": history_row.to_dict(),
        **model_metadata(samples_per_epoch),
    }
    torch.save(checkpoint, path)



def save_training_history(history: list[HistoryRow]) -> None:
    fieldnames = (
        list(history[0].to_dict().keys())
        if history
        else list(HistoryRow.__annotations__)
    )

    with SAVE.history_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in history)



def save_loss_curve(history: list[HistoryRow]) -> None:
    if not history:
        return

    plt.figure(figsize=(9, 6))
    plt.plot(
        [row.epoch for row in history],
        [row.total_loss for row in history],
        marker="o",
        markersize=3,
        label="Total",
    )
    plt.plot(
        [row.epoch for row in history],
        [row.final_l1 for row in history],
        marker="o",
        markersize=3,
        label="Final L1",
    )
    plt.plot(
        [row.epoch for row in history],
        [row.coarse_l1 for row in history],
        marker="o",
        markersize=3,
        label="Coarse L1",
    )
    plt.plot(
        [row.epoch for row in history],
        [1.0 - row.final_ssim for row in history],
        marker="o",
        markersize=3,
        label="Final SSIM Loss",
    )
    plt.plot(
        [row.epoch for row in history],
        [row.final_boundary for row in history],
        marker="o",
        markersize=3,
        label="Boundary Loss",
    )
    plt.plot(
        [row.epoch for row in history],
        [row.final_intensity for row in history],
        marker="o",
        markersize=3,
        label="Intensity Loss",
    )
    plt.plot(
        [row.epoch for row in history],
        [row.final_intensity_stats for row in history],
        marker="o",
        markersize=3,
        label="Intensity Stats Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(".1 PAM Intensity-Aware Structure-Preserving Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE.loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 13. Model smoke test
# ============================================================



def run_model_smoke_test(device: torch.device) -> None:
    """Fast shape/finite-value test before touching the full dataset."""

    model = ResidualEdgeWaveletMultiStageNet(
        in_channels=MODEL.in_channels,
        out_channels=MODEL.out_channels,
        base_channels=MODEL.base_channels,
        max_channels=MODEL.max_channels,
        use_edge_branch=MODEL.use_edge_branch,
        use_wavelet_branch=MODEL.use_wavelet_branch,
        use_stage2=MODEL.use_stage2,
    ).to(device)

    if TRAIN.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Smaller spatial shape keeps the smoke test fast while testing dynamic pad,
    # Haar DWT, encoder/decoder, both stages, and output contract.
    dummy = torch.randn(1, 3, 64, 128, device=device)

    if TRAIN.use_channels_last and device.type == "cuda":
        dummy = dummy.contiguous(memory_format=torch.channels_last)

    model.eval()
    with torch.inference_mode():
        output = model(dummy)
        details = model(dummy, return_intermediates=True)

    if output.shape != (1, 1, 64, 128):
        raise RuntimeError(
            f"Smoke test failed: output shape={tuple(output.shape)}"
        )

    if not torch.isfinite(output).all():
        raise RuntimeError("Smoke test failed: output contains NaN/Inf")

    required_keys = {
        "final",
        "coarse",
        "detail_gate",
        "refine_gate",
        "alpha",
        "beta",
        "gamma",
    }
    missing = required_keys - set(details)
    if missing:
        raise RuntimeError(f"Smoke test missing intermediate keys: {missing}")

    del model
    del dummy
    del output
    del details

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 14. Training
# ============================================================



def train() -> None:
    set_seed(SEED)
    configure_cuda()
    create_output_directories()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 92)
    print(".1 PAM ENHANCEMENT — INTENSITY-AWARE RESIDUAL + EDGE + WAVELET + 2-STAGE")
    print("=" * 92)
    print(f"Device                    : {device}")
    if device.type == "cuda":
        print(f"GPU                       : {torch.cuda.get_device_name(0)}")
    print(f"Train LOW                 : {DATA.train_low_dir}")
    print(f"Train HIGH                : {DATA.train_high_dir}")
    print(f"LOW shape                 : {DATA.adjacent_low_shape} [X,C,Y,Z]")
    print(f"HIGH shape                : {DATA.high_volume_shape} [X,Y,Z]")
    print(f"Input sample              : {INPUT_SAMPLE_SHAPE}")
    print(f"Target sample             : {TARGET_SAMPLE_SHAPE}")
    print(f"Epochs                    : {TRAIN.num_epochs}")
    print(f"Physical batch size       : {TRAIN.batch_size}")
    print(f"Gradient accumulation     : {TRAIN.gradient_accumulation_steps}")
    print(
        f"Effective batch size      : "
        f"{TRAIN.batch_size * TRAIN.gradient_accumulation_steps}"
    )
    print(f"Slices/volume/epoch       : {TRAIN.slices_per_volume_per_epoch}")
    print(f"Base / max channels       : {MODEL.base_channels} / {MODEL.max_channels}")
    print(f"Edge branch               : {MODEL.use_edge_branch}")
    print(f"Wavelet branch            : {MODEL.use_wavelet_branch}")
    print(f"Stage-2 refinement        : {MODEL.use_stage2}")
    print(
        "Loss                      : "
        f"{TRAIN.final_l1_weight:.2f}*L1 + "
        f"{TRAIN.final_ssim_weight:.2f}*SSIM + "
        f"{TRAIN.final_boundary_weight:.2f}*Boundary + "
        f"{TRAIN.final_intensity_weight:.2f}*Intensity + "
        f"{TRAIN.final_intensity_stats_weight:.2f}*IntensityStats + "
        f"{TRAIN.coarse_loss_weight:.2f}*Coarse"
    )
    print("Output activation         : None (Linear)")
    print("Prediction clamping       : None")
    print("=" * 92)

    print("\n[1/5] Running architecture smoke test...")
    run_model_smoke_test(device)
    print("      Smoke test passed.")

    print("\n[2/5] Building training dataset...")
    dataset = PAMAdjacentYZDataset(
        low_dir=DATA.train_low_dir,
        high_dir=DATA.train_high_dir,
    )

    sampler = EpochRandomXSliceSampler(
        dataset=dataset,
        slices_per_volume=TRAIN.slices_per_volume_per_epoch,
        seed=SEED,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": TRAIN.batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": TRAIN.num_workers,
        "pin_memory": TRAIN.pin_memory,
        "drop_last": False,
    }

    if TRAIN.num_workers > 0:
        loader_kwargs["persistent_workers"] = TRAIN.persistent_workers
        loader_kwargs["prefetch_factor"] = TRAIN.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    samples_per_epoch = len(sampler)

    print(f"      Volumes             : {dataset.num_volumes}")
    print(f"      Full dataset samples: {len(dataset):,}")
    print(f"      Samples/epoch       : {samples_per_epoch:,}")
    print(f"      Batches/epoch       : {len(loader):,}")

    print("\n[3/5] Building .1 model...")
    model = ResidualEdgeWaveletMultiStageNet(
        in_channels=MODEL.in_channels,
        out_channels=MODEL.out_channels,
        base_channels=MODEL.base_channels,
        max_channels=MODEL.max_channels,
        use_edge_branch=MODEL.use_edge_branch,
        use_wavelet_branch=MODEL.use_wavelet_branch,
        use_stage2=MODEL.use_stage2,
    ).to(device)

    if TRAIN.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_params, trainable_params = count_parameters(model)
    print(f"      Total parameters    : {total_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")

    criterion = MultiStageStructureLoss(
        final_l1_weight=TRAIN.final_l1_weight,
        final_ssim_weight=TRAIN.final_ssim_weight,
        final_boundary_weight=TRAIN.final_boundary_weight,
        final_intensity_weight=TRAIN.final_intensity_weight,
        final_intensity_stats_weight=TRAIN.final_intensity_stats_weight,
        coarse_weight=TRAIN.coarse_loss_weight,
        coarse_l1_weight=TRAIN.coarse_l1_weight,
        coarse_ssim_weight=TRAIN.coarse_ssim_weight,
        ssim_window_size=TRAIN.ssim_window_size,
        ssim_sigma=TRAIN.ssim_sigma,
        ssim_data_range=TRAIN.ssim_data_range,
        boundary_eps=TRAIN.boundary_eps,
        intensity_bright_boost=TRAIN.intensity_bright_boost,
        intensity_gamma=TRAIN.intensity_gamma,
        intensity_stats_std_weight=TRAIN.intensity_stats_std_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN.learning_rate,
        weight_decay=TRAIN.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN.num_epochs,
        eta_min=TRAIN.learning_rate * 0.05,
    )

    amp_enabled = TRAIN.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print("\n[4/5] Starting training...")

    history: list[HistoryRow] = []

    if SAVE.save_initial_map:
        print("      Saving initial MAP...")
        save_training_map(
            model=model,
            dataset=dataset,
            device=device,
            epoch=0,
        )

    for epoch in range(1, TRAIN.num_epochs + 1):
        epoch_start = time.time()
        sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_total = 0.0
        running_final_l1 = 0.0
        running_final_ssim = 0.0
        running_final_boundary = 0.0
        running_final_intensity = 0.0
        running_final_intensity_stats = 0.0
        running_coarse_l1 = 0.0
        running_coarse_ssim = 0.0
        seen_batches = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch:03d}/{TRAIN.num_epochs:03d}",
            dynamic_ncols=True,
        )

        accumulation_counter = 0

        for step, batch in enumerate(progress, start=1):
            inputs = batch["input"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            targets = batch["target"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if TRAIN.use_channels_last and device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(inputs, return_intermediates=True)
                total_loss, loss_parts = criterion(outputs, targets)

                scaled_loss = (
                    total_loss / TRAIN.gradient_accumulation_steps
                )

            scaler.scale(scaled_loss).backward()
            accumulation_counter += 1

            should_step = (
                accumulation_counter >= TRAIN.gradient_accumulation_steps
                or step == len(loader)
            )

            if should_step:
                scaler.unscale_(optimizer)

                if TRAIN.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=TRAIN.grad_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0

            running_total += float(total_loss.detach().item())
            running_final_l1 += float(loss_parts["final_l1"].item())
            running_final_ssim += float(loss_parts["final_ssim"].item())
            running_final_boundary += float(loss_parts["final_boundary"].item())
            running_final_intensity += float(
                loss_parts["final_intensity"].item()
            )
            running_final_intensity_stats += float(
                loss_parts["final_intensity_stats"].item()
            )
            running_coarse_l1 += float(loss_parts["coarse_l1"].item())
            running_coarse_ssim += float(loss_parts["coarse_ssim"].item())
            seen_batches += 1

            if step % TRAIN.log_every == 0 or step == len(loader):
                progress.set_postfix(
                    Total=f"{running_total / seen_batches:.6f}",
                    SSIM=f"{running_final_ssim / seen_batches:.4f}",
                    L1=f"{running_final_l1 / seen_batches:.5f}",
                    Int=f"{running_final_intensity / seen_batches:.5f}",
                    Bnd=f"{running_final_boundary / seen_batches:.5f}",
                    a=f"{float(model.alpha().detach().item()):.3f}",
                    b=f"{float(model.beta().detach().item()):.3f}",
                    g=f"{float(model.gamma().detach().item()):.3f}",
                )

        scheduler.step()

        epoch_seconds = time.time() - epoch_start
        denominator = max(seen_batches, 1)
        mean_total = running_total / denominator
        mean_final_l1 = running_final_l1 / denominator
        mean_final_ssim = running_final_ssim / denominator
        mean_final_boundary = running_final_boundary / denominator
        mean_final_intensity = running_final_intensity / denominator
        mean_final_intensity_stats = (
            running_final_intensity_stats / denominator
        )
        mean_coarse_l1 = running_coarse_l1 / denominator
        mean_coarse_ssim = running_coarse_ssim / denominator

        row = HistoryRow(
            epoch=epoch,
            total_loss=mean_total,
            final_l1=mean_final_l1,
            final_ssim=mean_final_ssim,
            final_boundary=mean_final_boundary,
            final_intensity=mean_final_intensity,
            final_intensity_stats=mean_final_intensity_stats,
            coarse_l1=mean_coarse_l1,
            coarse_ssim=mean_coarse_ssim,
            alpha=float(model.alpha().detach().item()),
            beta=float(model.beta().detach().item()),
            gamma=float(model.gamma().detach().item()),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            epoch_seconds=epoch_seconds,
        )
        history.append(row)

        print(
            f"Epoch {epoch:03d} finished | "
            f"Total={mean_total:.8f} | "
            f"Final L1={mean_final_l1:.8f} | "
            f"Final SSIM={mean_final_ssim:.6f} | "
            f"Boundary={mean_final_boundary:.8f} | "
            f"Intensity={mean_final_intensity:.8f} | "
            f"IntensityStats={mean_final_intensity_stats:.8f} | "
            f"Coarse L1={mean_coarse_l1:.8f} | "
            f"Coarse SSIM={mean_coarse_ssim:.6f} | "
            f"alpha={row.alpha:.4f} | "
            f"beta={row.beta:.4f} | "
            f"gamma={row.gamma:.4f} | "
            f"time={epoch_seconds / 60.0:.2f} min"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            history_row=row,
            samples_per_epoch=samples_per_epoch,
            path=SAVE.latest_model_path,
        )

        if epoch % SAVE.checkpoint_every == 0:
            checkpoint_path = (
                SAVE.checkpoint_dir / f"epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                history_row=row,
                samples_per_epoch=samples_per_epoch,
                path=checkpoint_path,
            )

        save_training_history(history)
        save_loss_curve(history)

        if epoch % SAVE.map_save_every == 0:
            print(f"      Saving MAP for epoch {epoch:03d}...")
            save_training_map(
                model=model,
                dataset=dataset,
                device=device,
                epoch=epoch,
            )

    print("\n[5/5] Saving final model...")

    final_row = history[-1]
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=TRAIN.num_epochs,
        history_row=final_row,
        samples_per_epoch=samples_per_epoch,
        path=SAVE.final_model_path,
    )

    print("=" * 92)
    print("TRAINING FINISHED")
    print(f"Final model : {SAVE.final_model_path}")
    print(f"Latest model: {SAVE.latest_model_path}")
    print(f"Checkpoints : {SAVE.checkpoint_dir}")
    print(f"Train MAPs  : {SAVE.train_map_dir}")
    print(f"History     : {SAVE.history_csv_path}")
    print("=" * 92)

    dataset.close()


# ============================================================
# 15. Entry point
# ============================================================


if __name__ == "__main__":
    train()
