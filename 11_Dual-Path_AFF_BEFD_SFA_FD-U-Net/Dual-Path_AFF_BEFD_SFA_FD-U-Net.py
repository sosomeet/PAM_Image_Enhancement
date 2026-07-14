from __future__ import annotations

import csv
import random
import re
import time
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("./data")
    header_size: int = 48
    dtype: type[np.float32] = np.float32
    high_volume_shape: tuple[int, int, int] = (200, 200, 512)  # [X, Y, Z]
    adjacent_low_shape: tuple[int, int, int, int] = (200, 3, 200, 512)  # [X, C, Y, Z]

    @property
    def train_low_dir(self) -> Path:
        return self.root / "3d_Train" / "LOW"

    @property
    def train_high_dir(self) -> Path:
        return self.root / "norm_Train" / "HIGH"

    @property
    def x_size(self) -> int:
        return self.high_volume_shape[0]

    @property
    def y_size(self) -> int:
        return self.high_volume_shape[1]

    @property
    def z_size(self) -> int:
        return self.high_volume_shape[2]

    @property
    def input_sample_shape(self) -> tuple[int, int, int]:
        return (self.adjacent_low_shape[1], self.y_size, self.z_size)

    @property
    def target_sample_shape(self) -> tuple[int, int, int]:
        return (1, self.y_size, self.z_size)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    num_epochs: int = 50
    learning_rate: float = 2e-4

    # Composite loss: 0.7 * L1 + 0.2 * SSIM + 0.1 * Edge
    l1_weight: float = 0.7
    ssim_weight: float = 0.2
    edge_weight: float = 0.1
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    ssim_data_range: float = 1.0
    edge_eps: float = 1e-6

    slices_per_volume_per_epoch: int = 32
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    use_amp: bool = True
    use_channels_last: bool = True
    log_every: int = 20

    @property
    def pin_memory(self) -> bool:
        return torch.cuda.is_available()

    @property
    def persistent_workers(self) -> bool:
        return self.num_workers > 0

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 3
    output_channels: int = 1
    base_channels: int = 8
    max_channels: int = 128
    dense_layers_per_block: int = 3

    sfa_dilation_rates: tuple[int, int, int] = (1, 3, 5)
    sfa_stages: tuple[str, ...] = ("encoder3", "encoder4", "bottleneck")

    edge_lambda_min: float = 0.8
    edge_lambda_max: float = 5.0
    edge_alpha: float = 2.0
    edge_beta: float = 1.0
    edge_eps: float = 1e-6
    befd_boundary_stages: tuple[str, ...] = ("encoder3", "encoder4", "bottleneck")
    befd_denoise_skip_stages: tuple[str, ...] = ("encoder1", "encoder2", "encoder3")
    denoise_max_positions: int = 512

    boundary_path_channels: tuple[int, int, int] = (8, 16, 32)
    boundary_fusion_stages: tuple[str, ...] = ("decoder4", "decoder3", "decoder1")
    boundary_gate_bias_init: float = -2.0

    aff_stages: tuple[str, ...] = ("decoder4", "decoder3", "decoder2", "decoder1")
    aff_reduction: int = 4
    aff_min_hidden_channels: int = 4
    aff_output_scale: float = 2.0

    pad_top: int = 4
    pad_bottom: int = 4


@dataclass(frozen=True)
class SaveConfig:
    experiment_root: Path = Path("./11_Dual-Path_AFF_BEFD_SFA_FD-U-Net")
    checkpoint_every: int = 10
    map_save_every: int = 5
    map_volume_index: int = 0
    map_batch_size: int = 16
    save_initial_map: bool = False

    @property
    def model_root(self) -> Path:
        return self.experiment_root / "models_enhancement"

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
        return self.experiment_root / "outputs"

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

    def create_directories(self) -> None:
        for directory in (
            self.latest_model_dir,
            self.checkpoint_dir,
            self.final_model_dir,
            self.train_map_dir,
            self.history_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DATA = DataConfig()
TRAIN = TrainConfig()
MODEL = ModelConfig()
SAVE = SaveConfig()

# Backward-compatible aliases for scripts that import the original constants.
SEED = TRAIN.seed
TRAIN_LOW_DIR = DATA.train_low_dir
TRAIN_HIGH_DIR = DATA.train_high_dir
HEADER_SIZE = DATA.header_size
DATA_DTYPE = DATA.dtype
HIGH_VOLUME_SHAPE = DATA.high_volume_shape
ADJACENT_LOW_SHAPE = DATA.adjacent_low_shape
X_SIZE, Y_SIZE, Z_SIZE = DATA.high_volume_shape
INPUT_CHANNELS = MODEL.input_channels
OUTPUT_CHANNELS = MODEL.output_channels
INPUT_SAMPLE_SHAPE = DATA.input_sample_shape
TARGET_SAMPLE_SHAPE = DATA.target_sample_shape
NUM_EPOCHS = TRAIN.num_epochs
LEARNING_RATE = TRAIN.learning_rate
L1_WEIGHT = TRAIN.l1_weight
SSIM_WEIGHT = TRAIN.ssim_weight
EDGE_WEIGHT = TRAIN.edge_weight
SSIM_WINDOW_SIZE = TRAIN.ssim_window_size
SSIM_SIGMA = TRAIN.ssim_sigma
SSIM_DATA_RANGE = TRAIN.ssim_data_range
EDGE_EPS = TRAIN.edge_eps
SLICES_PER_VOLUME_PER_EPOCH = TRAIN.slices_per_volume_per_epoch
BATCH_SIZE = TRAIN.batch_size
GRADIENT_ACCUMULATION_STEPS = TRAIN.gradient_accumulation_steps
NUM_WORKERS = TRAIN.num_workers
PREFETCH_FACTOR = TRAIN.prefetch_factor
PIN_MEMORY = TRAIN.pin_memory
PERSISTENT_WORKERS = TRAIN.persistent_workers
USE_AMP = TRAIN.use_amp
USE_CHANNELS_LAST = TRAIN.use_channels_last
LOG_EVERY = TRAIN.log_every
BASE_CHANNELS = MODEL.base_channels
MAX_CHANNELS = MODEL.max_channels
SFA_DILATION_RATES = MODEL.sfa_dilation_rates
SFA_STAGES = MODEL.sfa_stages
BEFD_EDGE_LAMBDA_MIN = MODEL.edge_lambda_min
BEFD_EDGE_LAMBDA_MAX = MODEL.edge_lambda_max
BEFD_EDGE_ALPHA = MODEL.edge_alpha
BEFD_EDGE_BETA = MODEL.edge_beta
BEFD_EDGE_EPS = MODEL.edge_eps
BEFD_BOUNDARY_STAGES = MODEL.befd_boundary_stages
BEFD_DENOISE_SKIP_STAGES = MODEL.befd_denoise_skip_stages
BEFD_DENOISE_MAX_POSITIONS = MODEL.denoise_max_positions
BOUNDARY_PATH_CHANNELS = MODEL.boundary_path_channels
BOUNDARY_FUSION_STAGES = MODEL.boundary_fusion_stages
BOUNDARY_GATE_BIAS_INIT = MODEL.boundary_gate_bias_init
AFF_STAGES = MODEL.aff_stages
AFF_REDUCTION = MODEL.aff_reduction
AFF_MIN_HIDDEN_CHANNELS = MODEL.aff_min_hidden_channels
AFF_OUTPUT_SCALE = MODEL.aff_output_scale
PAD_TOP = MODEL.pad_top
PAD_BOTTOM = MODEL.pad_bottom
CHECKPOINT_EVERY = SAVE.checkpoint_every
MAP_SAVE_EVERY = SAVE.map_save_every
MAP_VOLUME_INDEX = SAVE.map_volume_index
MAP_BATCH_SIZE = SAVE.map_batch_size
SAVE_INITIAL_MAP = SAVE.save_initial_map
MODEL_ROOT = SAVE.model_root
LATEST_MODEL_DIR = SAVE.latest_model_dir
CHECKPOINT_DIR = SAVE.checkpoint_dir
FINAL_MODEL_DIR = SAVE.final_model_dir
OUTPUT_ROOT = SAVE.output_root
TRAIN_MAP_DIR = SAVE.train_map_dir
HISTORY_DIR = SAVE.history_dir
LATEST_MODEL_PATH = SAVE.latest_model_path
FINAL_MODEL_PATH = SAVE.final_model_path
HISTORY_CSV_PATH = SAVE.history_csv_path
LOSS_CURVE_PATH = SAVE.loss_curve_path


# ============================================================
# Runtime utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_channels_last(device: torch.device) -> bool:
    return TRAIN.use_channels_last and device.type == "cuda"


def move_tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    channels_last: bool = False,
) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=True)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def autocast_context(device: torch.device, enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    gb = 1024**3
    print(f"\nCUDA Memory [{label}]")
    print(f"  Allocated : {torch.cuda.memory_allocated() / gb:.2f} GB")
    print(f"  Reserved  : {torch.cuda.memory_reserved() / gb:.2f} GB")
    print(f"  Peak      : {torch.cuda.max_memory_allocated() / gb:.2f} GB")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# File pairing and binary validation
# ============================================================


@dataclass(frozen=True)
class VolumePair:
    volume_key: str
    low_path: Path
    high_path: Path


def extract_volume_key(file_path: Path) -> str:
    return re.sub(r"_(LOW|HIGH)$", "", file_path.stem, flags=re.IGNORECASE)


def get_bin_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.bin"))
    if not files:
        raise RuntimeError(f"No .bin files found in: {directory}")
    return files


def validate_bin_file(
    file_path: Path,
    shape: tuple[int, ...],
    *,
    dtype: np.dtype[Any] | type[np.float32] = DATA.dtype,
    offset: int = DATA.header_size,
) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    np_dtype = np.dtype(dtype)
    expected_elements = int(np.prod(shape))
    expected_size = offset + expected_elements * np_dtype.itemsize
    actual_size = file_path.stat().st_size

    if actual_size != expected_size:
        raise ValueError(
            "\nBinary file size mismatch\n"
            f"File            : {file_path}\n"
            f"Actual bytes    : {actual_size:,}\n"
            f"Expected bytes  : {expected_size:,}\n"
            f"Expected shape  : {shape}\n"
            f"Expected dtype  : {np_dtype}\n"
            f"Header size     : {offset}"
        )


def build_file_pairs(low_dir: Path, high_dir: Path) -> list[VolumePair]:
    low_map = {extract_volume_key(path): path for path in get_bin_files(low_dir)}
    high_map = {extract_volume_key(path): path for path in get_bin_files(high_dir)}

    low_keys = set(low_map)
    high_keys = set(high_map)

    missing_high = sorted(low_keys - high_keys)
    missing_low = sorted(high_keys - low_keys)

    if missing_high:
        raise RuntimeError(
            "LOW files without matching HIGH files:\n" + "\n".join(missing_high)
        )
    if missing_low:
        raise RuntimeError(
            "HIGH files without matching LOW files:\n" + "\n".join(missing_low)
        )

    pairs: list[VolumePair] = []
    for key in sorted(low_keys & high_keys):
        low_path = low_map[key]
        high_path = high_map[key]
        validate_bin_file(low_path, DATA.adjacent_low_shape)
        validate_bin_file(high_path, DATA.high_volume_shape)
        pairs.append(VolumePair(key, low_path, high_path))

    if not pairs:
        raise RuntimeError("No valid LOW-HIGH pairs found.")
    return pairs


# ============================================================
# Dataset and sampler
# ============================================================


class PAMAdjacentYZDataset(Dataset[dict[str, Any]]):
    """
    Loads pre-generated 3-adjacent Y-Z LOW data.

    LOW : [X, C, Y, Z] = [200, 3, 200, 512]
    HIGH: [X, Y, Z]    = [200, 200, 512]

    One sample:
        input  = LOW[x]  -> [3, 200, 512]
        target = HIGH[x] -> [1, 200, 512]
    """

    def __init__(self, low_dir: Path, high_dir: Path) -> None:
        super().__init__()
        self.pairs = build_file_pairs(low_dir, high_dir)
        self.num_volumes = len(self.pairs)
        self.samples_per_volume = DATA.x_size
        self.total_samples = self.num_volumes * self.samples_per_volume
        self._low_memmaps: dict[int, np.memmap] = {}
        self._high_memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.total_samples

    def _open_memmap(
        self,
        volume_index: int,
        *,
        is_low: bool,
    ) -> np.memmap:
        cache = self._low_memmaps if is_low else self._high_memmaps
        if volume_index not in cache:
            pair = self.pairs[volume_index]
            path = pair.low_path if is_low else pair.high_path
            shape = DATA.adjacent_low_shape if is_low else DATA.high_volume_shape
            cache[volume_index] = np.memmap(
                filename=path,
                dtype=DATA.dtype,
                mode="r",
                offset=DATA.header_size,
                shape=shape,
                order="C",
            )
        return cache[volume_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(
                f"Dataset index out of range: {index}. Valid range: 0 ~ {len(self) - 1}"
            )

        volume_index, x_index = divmod(index, self.samples_per_volume)
        pair = self.pairs[volume_index]

        low_volume = self._open_memmap(volume_index, is_low=True)
        high_volume = self._open_memmap(volume_index, is_low=False)

        input_array = np.array(low_volume[x_index], dtype=np.float32, copy=True)
        target_array = np.array(high_volume[x_index], dtype=np.float32, copy=True)[None]

        self._validate_sample_shapes(
            input_array=input_array,
            target_array=target_array,
            pair=pair,
            x_index=x_index,
        )

        return {
            "input": torch.from_numpy(input_array),
            "target": torch.from_numpy(target_array),
            "volume_key": pair.volume_key,
            "volume_index": volume_index,
            "x_index": x_index,
        }

    @staticmethod
    def _validate_sample_shapes(
        *,
        input_array: np.ndarray,
        target_array: np.ndarray,
        pair: VolumePair,
        x_index: int,
    ) -> None:
        if input_array.shape != DATA.input_sample_shape:
            raise ValueError(
                "\nInput shape mismatch\n"
                f"Volume key    : {pair.volume_key}\n"
                f"X index       : {x_index}\n"
                f"Actual shape  : {input_array.shape}\n"
                f"Expected shape: {DATA.input_sample_shape}"
            )

        if target_array.shape != DATA.target_sample_shape:
            raise ValueError(
                "\nTarget shape mismatch\n"
                f"Volume key    : {pair.volume_key}\n"
                f"X index       : {x_index}\n"
                f"Actual shape  : {target_array.shape}\n"
                f"Expected shape: {DATA.target_sample_shape}"
            )

    def close(self) -> None:
        self._low_memmaps.clear()
        self._high_memmaps.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_low_memmaps"] = {}
        state["_high_memmaps"] = {}
        return state


class EpochRandomXSliceSampler(Sampler[int]):
    """Samples a deterministic random subset of X positions per volume each epoch."""

    def __init__(
        self,
        dataset: PAMAdjacentYZDataset,
        slices_per_volume: int,
        seed: int,
    ) -> None:
        if not 0 < slices_per_volume <= dataset.samples_per_volume:
            raise ValueError(
                f"slices_per_volume must be in [1, {dataset.samples_per_volume}], "
                f"got {slices_per_volume}"
            )

        self.dataset = dataset
        self.slices_per_volume = slices_per_volume
        self.seed = seed
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
            x_indices.sort()  # improves page-cache locality
            base_index = int(volume_index) * self.dataset.samples_per_volume
            global_indices.extend(base_index + int(x) for x in x_indices)

        return iter(global_indices)


def create_train_loader() -> tuple[
    PAMAdjacentYZDataset,
    EpochRandomXSliceSampler,
    DataLoader[dict[str, Any]],
]:
    dataset = PAMAdjacentYZDataset(DATA.train_low_dir, DATA.train_high_dir)
    sampler = EpochRandomXSliceSampler(
        dataset=dataset,
        slices_per_volume=TRAIN.slices_per_volume_per_epoch,
        seed=TRAIN.seed,
    )

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": TRAIN.batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": TRAIN.num_workers,
        "pin_memory": TRAIN.pin_memory,
        "drop_last": False,
        "persistent_workers": TRAIN.persistent_workers,
    }
    if TRAIN.num_workers > 0:
        kwargs["prefetch_factor"] = TRAIN.prefetch_factor

    return dataset, sampler, DataLoader(**kwargs)


# ============================================================
# Model blocks
# ============================================================


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        growth_rate: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        growth_rate = growth_rate or max(4, out_channels // 2)
        if growth_rate <= 0:
            raise ValueError(f"growth_rate must be > 0, got {growth_rate}")

        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate

        self.transition = nn.Conv2d(current_channels, out_channels, kernel_size=1)
        self.transition_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.transition_relu(self.transition(torch.cat(features, dim=1)))


class AdjacentScaleSelector(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")

        self.channels = channels
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=1),
        )

    def forward(
        self,
        feature_a: torch.Tensor,
        feature_b: torch.Tensor,
    ) -> torch.Tensor:
        if feature_a.shape != feature_b.shape:
            raise ValueError(
                f"Adjacent scale feature shape mismatch: {feature_a.shape} vs {feature_b.shape}"
            )

        batch, channels, height, width = feature_a.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        logits = self.weight_predictor(torch.cat([feature_a, feature_b], dim=1))
        weights = torch.softmax(logits.reshape(batch, 2, channels, height, width), dim=1)
        return weights[:, 0] * feature_a + weights[:, 1] * feature_b


class ScaleAwareFeatureAggregation(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation_rates: tuple[int, int, int] = MODEL.sfa_dilation_rates,
    ) -> None:
        super().__init__()
        if len(dilation_rates) != 3 or any(rate <= 0 for rate in dilation_rates):
            raise ValueError(f"SFA requires three positive dilation rates, got {dilation_rates}")

        self.channels = channels
        self.scale_convs = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
            )
            for rate in dilation_rates
        )
        self.scale_relu = nn.ReLU(inplace=True)
        self.select_12 = AdjacentScaleSelector(channels)
        self.select_23 = AdjacentScaleSelector(channels)
        self.output_fusion = nn.Conv2d(channels, channels, kernel_size=1)
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.size(1)}")

        d1, d2, d3 = [self.scale_relu(conv(x)) for conv in self.scale_convs]
        fused = self.select_12(d1, d2) + self.select_23(d2, d3) + x
        return self.output_relu(self.output_fusion(fused))


class SobelBoundaryAttention(nn.Module):
    def __init__(
        self,
        lambda_min: float = MODEL.edge_lambda_min,
        lambda_max: float = MODEL.edge_lambda_max,
        alpha: float = MODEL.edge_alpha,
        beta: float = MODEL.edge_beta,
        eps: float = MODEL.edge_eps,
    ) -> None:
        super().__init__()
        if lambda_max <= lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer("sobel_y", sobel_y, persistent=True)

    def forward(self, center: torch.Tensor) -> torch.Tensor:
        if center.ndim != 4 or center.size(1) != 1:
            raise ValueError(
                f"Expected center plane [B, 1, H, W], got {tuple(center.shape)}"
            )

        center_float = center.float()
        padded = F.pad(center_float, pad=(1, 1, 1, 1), mode="reflect")
        gradient_x = F.conv2d(padded, self.sobel_x)
        gradient_y = F.conv2d(padded, self.sobel_y)
        gradient = torch.sqrt(gradient_x.square() + gradient_y.square() + self.eps)

        in_range = (gradient >= self.lambda_min) & (gradient <= self.lambda_max)
        normalized = (gradient - self.lambda_min) / (self.lambda_max - self.lambda_min)
        transformed = (1.0 - normalized) * self.alpha + self.beta
        attention = torch.where(in_range, transformed, torch.ones_like(transformed))
        return attention.to(dtype=center.dtype)


class FeatureDenoisingBlock(nn.Module):
    def __init__(self, channels: int, max_positions: int = MODEL.denoise_max_positions) -> None:
        super().__init__()
        if channels <= 0 or max_positions <= 0:
            raise ValueError("channels and max_positions must be > 0")

        self.channels = channels
        self.max_positions = max_positions
        self.output_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def _get_pooled_size(self, height: int, width: int) -> tuple[int, int]:
        total_positions = height * width
        if total_positions <= self.max_positions:
            return height, width

        scale = (self.max_positions / total_positions) ** 0.5
        pooled_height = max(1, int(height * scale))
        pooled_width = max(1, int(width * scale))

        while pooled_height * pooled_width > self.max_positions:
            if pooled_width >= pooled_height and pooled_width > 1:
                pooled_width -= 1
            elif pooled_height > 1:
                pooled_height -= 1
            else:
                break

        return pooled_height, pooled_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(x.shape)}")

        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        pooled_height, pooled_width = self._get_pooled_size(height, width)
        pooled = (
            x
            if (pooled_height, pooled_width) == (height, width)
            else F.adaptive_avg_pool2d(x, (pooled_height, pooled_width))
        )

        positions = pooled.float().flatten(2).transpose(1, 2)  # [B, N, C]
        affinity = torch.softmax(torch.bmm(positions, positions.transpose(1, 2)), dim=-1)
        denoised = torch.bmm(affinity, positions)
        denoised = denoised.transpose(1, 2).reshape(
            batch, channels, pooled_height, pooled_width
        )
        denoised = self.output_conv(denoised.to(dtype=x.dtype))

        if (pooled_height, pooled_width) != (height, width):
            denoised = F.interpolate(
                denoised,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        return x + denoised


class BoundaryConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BoundaryDetailPath(nn.Module):
    def __init__(self, c1: int, c2: int, c3: int, c4: int) -> None:
        super().__init__()
        self.block1 = BoundaryConvBlock(2, c1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = BoundaryConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = BoundaryConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.project4 = nn.Sequential(nn.Conv2d(c3, c4, kernel_size=1), nn.ReLU(inplace=True))

    def forward(
        self,
        center: torch.Tensor,
        edge_guidance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if center.shape != edge_guidance.shape:
            raise ValueError(
                f"Boundary input mismatch: center={tuple(center.shape)}, "
                f"edge={tuple(edge_guidance.shape)}"
            )

        boundary_input = torch.cat([center, edge_guidance], dim=1)
        b1 = self.block1(boundary_input)
        b2 = self.block2(self.pool1(b1))
        b3 = self.block3(self.pool2(b2))
        b4 = self.project4(self.pool3(b3))
        return b1, b2, b3, b4


class GatedBoundaryFusion(nn.Module):
    def __init__(
        self,
        main_channels: int,
        boundary_channels: int,
        gate_bias_init: float = MODEL.boundary_gate_bias_init,
    ) -> None:
        super().__init__()
        self.main_channels = main_channels
        self.boundary_channels = boundary_channels

        self.boundary_projection = nn.Conv2d(boundary_channels, main_channels, kernel_size=1)
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(main_channels * 2, main_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(main_channels, main_channels, kernel_size=1),
        )
        nn.init.constant_(self.gate_predictor[-1].bias, float(gate_bias_init))

    def forward(
        self,
        main_feature: torch.Tensor,
        boundary_feature: torch.Tensor,
    ) -> torch.Tensor:
        if main_feature.ndim != 4 or boundary_feature.ndim != 4:
            raise ValueError("GatedBoundaryFusion expects 4D tensors")
        if main_feature.size(1) != self.main_channels:
            raise ValueError(f"Expected main channels={self.main_channels}, got {main_feature.size(1)}")
        if boundary_feature.size(1) != self.boundary_channels:
            raise ValueError(
                f"Expected boundary channels={self.boundary_channels}, got {boundary_feature.size(1)}"
            )

        if boundary_feature.shape[-2:] != main_feature.shape[-2:]:
            boundary_feature = F.interpolate(
                boundary_feature,
                size=main_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        projected_boundary = self.boundary_projection(boundary_feature)
        gate = torch.sigmoid(
            self.gate_predictor(torch.cat([main_feature, projected_boundary], dim=1))
        )
        return main_feature + gate * projected_boundary


class AdaptiveFeatureFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = MODEL.aff_reduction,
        min_hidden_channels: int = MODEL.aff_min_hidden_channels,
        output_scale: float = MODEL.aff_output_scale,
    ) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0 or min_hidden_channels <= 0 or output_scale <= 0:
            raise ValueError("AFF configuration values must be > 0")

        self.channels = channels
        self.output_scale = float(output_scale)
        hidden_channels = max(min_hidden_channels, channels // reduction)

        self.decoder_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip_projection = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.dirac_(self.decoder_projection.weight)
        nn.init.zeros_(self.decoder_projection.bias)
        nn.init.dirac_(self.skip_projection.weight)
        nn.init.zeros_(self.skip_projection.bias)

        self.local_context = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1),
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1),
        )

        for final_layer in (self.local_context[-1], self.global_context[-1]):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        skip_feature: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_feature.ndim != 4 or skip_feature.ndim != 4:
            raise ValueError("AdaptiveFeatureFusion expects 4D tensors")
        if decoder_feature.size(1) != self.channels or skip_feature.size(1) != self.channels:
            raise ValueError(
                f"AFF expects {self.channels} channels, got "
                f"decoder={decoder_feature.size(1)}, skip={skip_feature.size(1)}"
            )

        if skip_feature.shape[-2:] != decoder_feature.shape[-2:]:
            skip_feature = F.interpolate(
                skip_feature,
                size=decoder_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        decoder_projected = self.decoder_projection(decoder_feature)
        skip_projected = self.skip_projection(skip_feature)
        joint_context = decoder_projected + skip_projected
        logits = self.local_context(joint_context) + self.global_context(joint_context)

        batch, _, height, width = logits.shape
        weights = torch.softmax(
            logits.reshape(batch, 2, self.channels, height, width),
            dim=1,
        )

        weighted_decoder = self.output_scale * weights[:, 0] * decoder_projected
        weighted_skip = self.output_scale * weights[:, 1] * skip_projected
        return torch.cat([weighted_decoder, weighted_skip], dim=1)


class DualPathFullyDenseSFABEFDAFFUNet(nn.Module):
    """
    E5 model for 3-adjacent Y-Z PAM enhancement.

    Main path:
        Fully Dense U-Net + SFA + BEFD feature denoising
    Detail path:
        Center LOW + Sobel guidance
    Decoder:
        Gated boundary fusion + AFF
    """

    def __init__(
        self,
        in_channels: int = MODEL.input_channels,
        out_channels: int = MODEL.output_channels,
        base_channels: int = MODEL.base_channels,
        dense_layers_per_block: int = MODEL.dense_layers_per_block,
        sfa_dilation_rates: tuple[int, int, int] = MODEL.sfa_dilation_rates,
        edge_lambda_min: float = MODEL.edge_lambda_min,
        edge_lambda_max: float = MODEL.edge_lambda_max,
        edge_alpha: float = MODEL.edge_alpha,
        edge_beta: float = MODEL.edge_beta,
        denoise_max_positions: int = MODEL.denoise_max_positions,
        gate_bias_init: float = MODEL.boundary_gate_bias_init,
        aff_reduction: int = MODEL.aff_reduction,
        aff_min_hidden_channels: int = MODEL.aff_min_hidden_channels,
        aff_output_scale: float = MODEL.aff_output_scale,
    ) -> None:
        super().__init__()

        c1, c2, c3, c4, c5 = [base_channels * (2**i) for i in range(5)]

        # Preserve the original public attributes for checkpoint/evaluation compatibility.
        self.dense_layers_per_block = dense_layers_per_block
        self.sfa_dilation_rates = tuple(sfa_dilation_rates)
        self.denoise_max_positions = int(denoise_max_positions)
        self.gate_bias_init = float(gate_bias_init)
        self.aff_reduction = int(aff_reduction)
        self.aff_min_hidden_channels = int(aff_min_hidden_channels)
        self.aff_output_scale = float(aff_output_scale)

        if c5 != MODEL.max_channels:
            raise ValueError(
                f"Channel configuration mismatch: bottleneck={c5}, max={MODEL.max_channels}"
            )

        self.boundary_attention = SobelBoundaryAttention(
            lambda_min=edge_lambda_min,
            lambda_max=edge_lambda_max,
            alpha=edge_alpha,
            beta=edge_beta,
        )
        self.boundary_path = BoundaryDetailPath(c1, c2, c3, c4)

        self.encoder1 = DenseBlock(in_channels, c1, dense_layers_per_block)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = DenseBlock(c1, c2, dense_layers_per_block)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = DenseBlock(c2, c3, dense_layers_per_block)
        self.sfa3 = ScaleAwareFeatureAggregation(c3, sfa_dilation_rates)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = DenseBlock(c3, c4, dense_layers_per_block)
        self.sfa4 = ScaleAwareFeatureAggregation(c4, sfa_dilation_rates)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = DenseBlock(c4, c5, dense_layers_per_block)
        self.sfa_bottleneck = ScaleAwareFeatureAggregation(c5, sfa_dilation_rates)

        self.denoise1 = FeatureDenoisingBlock(c1, denoise_max_positions)
        self.denoise2 = FeatureDenoisingBlock(c2, denoise_max_positions)
        self.denoise3 = FeatureDenoisingBlock(c3, denoise_max_positions)

        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.boundary_fusion4 = GatedBoundaryFusion(c4, c4, gate_bias_init)
        self.aff4 = AdaptiveFeatureFusion(
            c4, aff_reduction, aff_min_hidden_channels, aff_output_scale
        )
        self.decoder4 = DenseBlock(c4 * 2, c4, dense_layers_per_block)

        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.boundary_fusion3 = GatedBoundaryFusion(c3, c3, gate_bias_init)
        self.aff3 = AdaptiveFeatureFusion(
            c3, aff_reduction, aff_min_hidden_channels, aff_output_scale
        )
        self.decoder3 = DenseBlock(c3 * 2, c3, dense_layers_per_block)

        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.aff2 = AdaptiveFeatureFusion(
            c2, aff_reduction, aff_min_hidden_channels, aff_output_scale
        )
        self.decoder2 = DenseBlock(c2 * 2, c2, dense_layers_per_block)

        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.boundary_fusion1 = GatedBoundaryFusion(c1, c1, gate_bias_init)
        self.aff1 = AdaptiveFeatureFusion(
            c1, aff_reduction, aff_min_hidden_channels, aff_output_scale
        )
        self.decoder1 = DenseBlock(c1 * 2, c1, dense_layers_per_block)

        self.output_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            pad=(0, 0, MODEL.pad_top, MODEL.pad_bottom),
            mode="reflect",
        )

        center = x[:, 1:2]
        edge_attention = self.boundary_attention(center)
        edge_guidance = (edge_attention - 1.0).clamp_min(0.0)
        b1, _, b3, b4 = self.boundary_path(center, edge_guidance)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))

        e3 = self.sfa3(self.encoder3(self.pool2(e2)))
        e3 = e3 * self._resize_attention(edge_attention, e3)

        e4 = self.sfa4(self.encoder4(self.pool3(e3)))
        e4 = e4 * self._resize_attention(edge_attention, e4)

        bottleneck = self.sfa_bottleneck(self.bottleneck(self.pool4(e4)))
        bottleneck = bottleneck * self._resize_attention(edge_attention, bottleneck)

        skip1 = self.denoise1(e1)
        skip2 = self.denoise2(e2)
        skip3 = self.denoise3(e3)

        d4 = self.boundary_fusion4(self.upconv4(bottleneck), b4)
        d4 = self.decoder4(self.aff4(d4, e4))

        d3 = self.boundary_fusion3(self.upconv3(d4), b3)
        d3 = self.decoder3(self.aff3(d3, skip3))

        d2 = self.upconv2(d3)
        d2 = self.decoder2(self.aff2(d2, skip2))

        d1 = self.boundary_fusion1(self.upconv1(d2), b1)
        d1 = self.decoder1(self.aff1(d1, skip1))

        output = self.output_conv(d1)
        return output[:, :, MODEL.pad_top : -MODEL.pad_bottom, :]


# ============================================================
# Composite loss: L1 + SSIM + Edge
# ============================================================


def _gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"SSIM window_size must be a positive odd integer, got {window_size}")
    if sigma <= 0:
        raise ValueError(f"SSIM sigma must be > 0, got {sigma}")

    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


class SSIMLoss(nn.Module):
    """Differentiable 2D SSIM loss: 1 - mean(SSIM)."""

    def __init__(
        self,
        window_size: int = TRAIN.ssim_window_size,
        sigma: float = TRAIN.ssim_sigma,
        data_range: float = TRAIN.ssim_data_range,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        if data_range <= 0:
            raise ValueError(f"SSIM data_range must be > 0, got {data_range}")

        gaussian_1d = _gaussian_1d(window_size, sigma)
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
        self.register_buffer(
            "window",
            gaussian_2d.reshape(1, 1, window_size, window_size),
            persistent=True,
        )
        self.window_size = int(window_size)
        self.data_range = float(data_range)
        self.c1 = float((k1 * data_range) ** 2)
        self.c2 = float((k2 * data_range) ** 2)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"SSIM shape mismatch: prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )
        if prediction.ndim != 4:
            raise ValueError(f"SSIM expects [B, C, H, W], got {tuple(prediction.shape)}")

        channels = prediction.size(1)
        window = self.window.to(device=prediction.device, dtype=prediction.dtype)
        window = window.expand(channels, 1, self.window_size, self.window_size)
        padding = self.window_size // 2

        mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
        mu_y = F.conv2d(target, window, padding=padding, groups=channels)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(
            prediction.square(), window, padding=padding, groups=channels
        ) - mu_x_sq
        sigma_y_sq = F.conv2d(
            target.square(), window, padding=padding, groups=channels
        ) - mu_y_sq
        sigma_xy = F.conv2d(
            prediction * target, window, padding=padding, groups=channels
        ) - mu_xy

        numerator = (2.0 * mu_xy + self.c1) * (2.0 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )

        ssim_map = numerator / denominator.clamp_min(1e-12)
        return 1.0 - ssim_map.mean()


class SobelEdgeLoss(nn.Module):
    """L1 difference between Sobel gradient magnitudes of prediction and target."""

    def __init__(self, eps: float = TRAIN.edge_eps) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"Edge eps must be > 0, got {eps}")

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=True)
        self.register_buffer("sobel_y", sobel_y, persistent=True)
        self.eps = float(eps)

    def _gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.size(1)
        sobel_x = self.sobel_x.to(device=image.device, dtype=image.dtype).expand(
            channels, 1, 3, 3
        )
        sobel_y = self.sobel_y.to(device=image.device, dtype=image.dtype).expand(
            channels, 1, 3, 3
        )

        padded = F.pad(image, pad=(1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, sobel_x, groups=channels)
        grad_y = F.conv2d(padded, sobel_y, groups=channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Edge loss shape mismatch: prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )

        pred_edges = self._gradient_magnitude(prediction)
        target_edges = self._gradient_magnitude(target)
        return F.l1_loss(pred_edges, target_edges)


@dataclass
class LossOutput:
    total: torch.Tensor
    l1: torch.Tensor
    ssim: torch.Tensor
    edge: torch.Tensor


class PAMCompositeLoss(nn.Module):
    """
    Total loss = 0.7 * L1 + 0.2 * SSIM Loss + 0.1 * Sobel Edge Loss.
    """

    def __init__(self) -> None:
        super().__init__()
        weights = TRAIN.l1_weight + TRAIN.ssim_weight + TRAIN.edge_weight
        if not np.isclose(weights, 1.0):
            raise ValueError(
                f"Composite loss weights must sum to 1.0, got {weights:.6f}"
            )

        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.edge = SobelEdgeLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> LossOutput:
        l1_loss = self.l1(prediction, target)
        ssim_loss = self.ssim(prediction, target)
        edge_loss = self.edge(prediction, target)

        total_loss = (
            TRAIN.l1_weight * l1_loss
            + TRAIN.ssim_weight * ssim_loss
            + TRAIN.edge_weight * edge_loss
        )
        return LossOutput(
            total=total_loss,
            l1=l1_loss,
            ssim=ssim_loss,
            edge=edge_loss,
        )


# ============================================================
# Verification and optimization
# ============================================================


@torch.no_grad()
def verify_model_shape(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    device: torch.device,
    amp_enabled: bool,
) -> None:
    batch = next(iter(train_loader))
    channels_last = use_channels_last(device)
    inputs = move_tensor_to_device(batch["input"], device, channels_last=channels_last)
    targets = move_tensor_to_device(batch["target"], device)

    previous_mode = model.training
    model.eval()
    with autocast_context(device, amp_enabled):
        outputs = model(inputs)
    model.train(previous_mode)

    print("\n" + "=" * 70)
    print("MODEL SHAPE VERIFICATION")
    print("=" * 70)
    print(f"Input shape : {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")

    if outputs.shape != targets.shape:
        raise ValueError(
            "\nModel output and target shape mismatch.\n"
            f"Output: {tuple(outputs.shape)}\n"
            f"Target: {tuple(targets.shape)}"
        )
    print("\nModel output shape is valid.")


def create_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    kwargs = {
        "params": model.parameters(),
        "lr": TRAIN.learning_rate,
        "betas": (0.9, 0.999),
    }

    if device.type == "cuda":
        try:
            optimizer = torch.optim.Adam(**kwargs, fused=True)
            print("  Adam implementation    : fused")
            return optimizer
        except (TypeError, RuntimeError):
            pass

    print("  Adam implementation    : standard")
    return torch.optim.Adam(**kwargs)


# ============================================================
# Training
# ============================================================


@dataclass
class EpochLossMetrics:
    total: float
    l1: float
    ssim: float
    edge: float


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[dict[str, Any]],
    criterion: PAMCompositeLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    amp_enabled: bool,
) -> EpochLossMetrics:
    model.train()
    channels_last = use_channels_last(device)
    running_total = torch.zeros((), device=device, dtype=torch.float64)
    running_l1 = torch.zeros((), device=device, dtype=torch.float64)
    running_ssim = torch.zeros((), device=device, dtype=torch.float64)
    running_edge = torch.zeros((), device=device, dtype=torch.float64)
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch:03d}/{total_epochs:03d}",
        leave=True,
        mininterval=0.5,
    )

    for step, batch in progress_bar:
        inputs = move_tensor_to_device(batch["input"], device, channels_last=channels_last)
        targets = move_tensor_to_device(batch["target"], device)
        batch_size = inputs.size(0)

        with autocast_context(device, amp_enabled):
            predictions = model(inputs)
            loss_output = criterion(predictions, targets)
            loss_for_backward = loss_output.total / TRAIN.gradient_accumulation_steps

        scaler.scale(loss_for_backward).backward()

        is_accumulation_boundary = (
            (step + 1) % TRAIN.gradient_accumulation_steps == 0
        )
        is_last_batch = step + 1 == len(train_loader)

        if is_accumulation_boundary or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_total += loss_output.total.detach().double() * batch_size
        running_l1 += loss_output.l1.detach().double() * batch_size
        running_ssim += loss_output.ssim.detach().double() * batch_size
        running_edge += loss_output.edge.detach().double() * batch_size
        total_samples += batch_size

        if (step + 1) % TRAIN.log_every == 0 or is_last_batch:
            progress_bar.set_postfix(
                Total=f"{(running_total / total_samples).item():.6f}",
                L1=f"{(running_l1 / total_samples).item():.6f}",
                SSIM=f"{(running_ssim / total_samples).item():.6f}",
                Edge=f"{(running_edge / total_samples).item():.6f}",
            )

    return EpochLossMetrics(
        total=(running_total / total_samples).item(),
        l1=(running_l1 / total_samples).item(),
        ssim=(running_ssim / total_samples).item(),
        edge=(running_edge / total_samples).item(),
    )


# ============================================================
# MAP visualization
# ============================================================


def _build_volume_loader(
    dataset: PAMAdjacentYZDataset,
    volume_index: int,
) -> DataLoader[dict[str, Any]]:
    if not 0 <= volume_index < dataset.num_volumes:
        raise IndexError(
            f"Invalid MAP volume index: {volume_index}. "
            f"Valid range: 0 ~ {dataset.num_volumes - 1}"
        )

    start = volume_index * dataset.samples_per_volume
    end = start + dataset.samples_per_volume
    subset = Subset(dataset, range(start, end))

    return DataLoader(
        subset,
        batch_size=SAVE.map_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=TRAIN.pin_memory,
        drop_last=False,
    )


def _validate_reconstructed_tensor(label: str, tensor: torch.Tensor) -> None:
    expected = (DATA.x_size, 1, DATA.y_size, DATA.z_size)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"\n{label} reconstruction shape mismatch.\n"
            f"Actual   : {tuple(tensor.shape)}\n"
            f"Expected : {expected}"
        )


def _safe_percentile_max(array: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    value = float(np.percentile(array, percentile))
    if not np.isfinite(value) or value <= 0:
        value = float(array.max()) if array.size else fallback
    return value if value > 0 else fallback


@torch.no_grad()
def save_epoch_map(
    model: nn.Module,
    train_dataset: PAMAdjacentYZDataset,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    volume_index: int = SAVE.map_volume_index,
) -> None:
    previous_mode = model.training
    model.eval()
    channels_last = use_channels_last(device)
    volume_loader = _build_volume_loader(train_dataset, volume_index)

    low_slices: list[torch.Tensor] = []
    pred_slices: list[torch.Tensor] = []
    target_slices: list[torch.Tensor] = []
    volume_key: str | None = None

    try:
        for batch in volume_loader:
            inputs = move_tensor_to_device(batch["input"], device, channels_last=channels_last)
            targets = batch["target"]

            with autocast_context(device, amp_enabled):
                predictions = model(inputs)

            low_slices.append(inputs[:, 1:2].detach().float().cpu())
            pred_slices.append(predictions.clamp_min(0.0).detach().float().cpu())
            target_slices.append(targets.detach().float().cpu())

            if volume_key is None:
                volume_key = batch["volume_key"][0]

        low_tensor = torch.cat(low_slices, dim=0)
        pred_tensor = torch.cat(pred_slices, dim=0)
        target_tensor = torch.cat(target_slices, dim=0)

        for label, tensor in (
            ("LOW", low_tensor),
            ("Prediction", pred_tensor),
            ("Target", target_tensor),
        ):
            _validate_reconstructed_tensor(label, tensor)

        low_volume = low_tensor[:, 0].numpy()
        pred_volume = pred_tensor[:, 0].numpy()
        target_volume = target_tensor[:, 0].numpy()

        for label, volume in (
            ("LOW", low_volume),
            ("Prediction", pred_volume),
            ("Target", target_volume),
        ):
            if tuple(volume.shape) != DATA.high_volume_shape:
                raise ValueError(
                    f"\n{label} full-volume shape mismatch.\n"
                    f"Actual   : {tuple(volume.shape)}\n"
                    f"Expected : {DATA.high_volume_shape}"
                )

        low_map = np.max(low_volume, axis=2)
        pred_map = np.max(pred_volume, axis=2)
        target_map = np.max(target_volume, axis=2)
        error_map = np.abs(pred_map - target_map)

        display_vmax = _safe_percentile_max(target_map, 99.5)
        error_vmax = _safe_percentile_max(error_map, 99.5)

        figure, axes = plt.subplots(1, 4, figsize=(20, 5))
        panels = (
            (low_map, "LOW MAP", "hot", 0.0, display_vmax),
            (pred_map, f"Prediction MAP\nEpoch {epoch}", "hot", 0.0, display_vmax),
            (target_map, "HIGH MAP", "hot", 0.0, display_vmax),
            (error_map, "Absolute Error", "viridis", 0.0, error_vmax),
        )

        for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
            shown = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title)
            axis.axis("off")
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)

        figure.suptitle(
            "Dual-Path Fully Dense + SFA + BEFD + AFF U-Net Train MAP | "
            f"{volume_key} | Epoch {epoch}",
            fontsize=14,
        )
        figure.tight_layout()

        output_path = SAVE.train_map_dir / f"epoch_{epoch:03d}_map.png"
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved Train MAP: {output_path}")
    finally:
        model.train(previous_mode)


# ============================================================
# Saving and history
# ============================================================


@dataclass
class HistoryRow:
    epoch: int
    train_total_loss: float
    train_l1_loss: float
    train_ssim_loss: float
    train_edge_loss: float
    epoch_time_seconds: float
    peak_gpu_memory_gb: float
    samples_this_epoch: int


def build_checkpoint_metadata(samples_per_epoch: int) -> dict[str, Any]:
    return {
        "model_name": "DualPathFullyDenseSFABEFDAFFUNet",
        "experiment": "E5",
        "training_plane": "YZ",
        "adjacent_axis": "X",
        "low_data_shape_order": "[X, C, Y, Z]",
        "high_data_shape_order": "[X, Y, Z]",
        "high_target_definition": "HIGH[x, :, :]",
        "input_channels": MODEL.input_channels,
        "output_channels": MODEL.output_channels,
        "base_channels": MODEL.base_channels,
        "max_channels": MODEL.max_channels,
        "dense_layers_per_block": MODEL.dense_layers_per_block,
        "dense_growth_rule": "max(4, out_channels // 2)",
        "dense_transition": "1x1 Conv + ReLU",
        "sfa_enabled": True,
        "sfa_stages": list(MODEL.sfa_stages),
        "sfa_dilation_rates": list(MODEL.sfa_dilation_rates),
        "sfa_adjacent_pairs": ["1-3", "3-5"],
        "sfa_dynamic_selection": "per-channel per-pixel softmax over adjacent scales",
        "sfa_residual_fusion": True,
        "befd_enabled": True,
        "befd_boundary_stages": list(MODEL.befd_boundary_stages),
        "befd_boundary_source": "center LOW channel (X)",
        "befd_edge_detector": "fixed Sobel x/y magnitude",
        "befd_edge_lambda_min": MODEL.edge_lambda_min,
        "befd_edge_lambda_max": MODEL.edge_lambda_max,
        "befd_edge_alpha": MODEL.edge_alpha,
        "befd_edge_beta": MODEL.edge_beta,
        "befd_feature_denoising_stages": list(MODEL.befd_denoise_skip_stages),
        "befd_denoising_type": "pooled dot-product non-local attention + 1x1 Conv + residual",
        "befd_denoise_max_positions": MODEL.denoise_max_positions,
        "dual_path_enabled": True,
        "dual_path_main_path": "3-adjacent Fully Dense U-Net + SFA + BEFD",
        "dual_path_boundary_input": "center LOW + BEFD Sobel guidance",
        "dual_path_boundary_channels": list(MODEL.boundary_path_channels),
        "dual_path_boundary_fusion_stages": list(MODEL.boundary_fusion_stages),
        "dual_path_fusion_type": "per-channel per-pixel sigmoid gated residual injection",
        "dual_path_gate_bias_init": MODEL.boundary_gate_bias_init,
        "aff_enabled": True,
        "aff_stages": list(MODEL.aff_stages),
        "aff_type": "SCS-Net-inspired local+global adaptive hierarchical fusion",
        "aff_weighting": "two-branch per-channel per-pixel softmax",
        "aff_reduction": MODEL.aff_reduction,
        "aff_min_hidden_channels": MODEL.aff_min_hidden_channels,
        "aff_output_scale": MODEL.aff_output_scale,
        "aff_output": "weighted concatenation preserving 2C decoder input",
        "aff_initialization": "zero final logits -> equal 0.5/0.5 branch weights",
        "physical_batch_size": TRAIN.batch_size,
        "gradient_accumulation_steps": TRAIN.gradient_accumulation_steps,
        "effective_batch_size": TRAIN.effective_batch_size,
        "slices_per_volume_per_epoch": TRAIN.slices_per_volume_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "use_amp": TRAIN.use_amp,
        "use_channels_last": TRAIN.use_channels_last,
        "final_output_activation": "None (Linear)",
        "loss": "0.7*L1 + 0.2*(1-SSIM) + 0.1*SobelEdge",
        "l1_weight": TRAIN.l1_weight,
        "ssim_weight": TRAIN.ssim_weight,
        "edge_weight": TRAIN.edge_weight,
        "ssim_window_size": TRAIN.ssim_window_size,
        "ssim_sigma": TRAIN.ssim_sigma,
        "ssim_data_range": TRAIN.ssim_data_range,
        "edge_operator": "Sobel gradient magnitude L1",
        "optimizer": "Adam",
        "learning_rate": TRAIN.learning_rate,
        "map_save_every": SAVE.map_save_every,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    train_loss: float,
    samples_per_epoch: int,
    path: Path,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        **build_checkpoint_metadata(samples_per_epoch),
    }
    torch.save(checkpoint, path)


def save_training_history(history: list[HistoryRow]) -> None:
    fieldnames = list(asdict(history[0]).keys()) if history else list(HistoryRow.__annotations__)
    with SAVE.history_csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in history)


def save_loss_curve(history: list[HistoryRow]) -> None:
    if not history:
        return

    epochs = [row.epoch for row in history]
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, [row.train_total_loss for row in history], label="Total")
    plt.plot(epochs, [row.train_l1_loss for row in history], label="L1")
    plt.plot(epochs, [row.train_ssim_loss for row in history], label="SSIM Loss")
    plt.plot(epochs, [row.train_edge_loss for row in history], label="Edge Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("E5 Training Loss: L1 + SSIM + Sobel Edge")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE.loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Console summaries
# ============================================================


def print_experiment_header(device: torch.device) -> None:
    print("\n" + "=" * 70)
    print("PAM 3-ADJACENT Y-Z DUAL-PATH FULLY DENSE + SFA + BEFD + AFF U-NET TRAINING")
    print("TRAIN-ONLY | PRE-GENERATED 3-ADJACENT LOW DATA")
    print("=" * 70)
    print(f"Device                  : {device}")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU                     : {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory        : {props.total_memory / (1024**3):.2f} GB")


def print_data_summary(
    dataset: PAMAdjacentYZDataset,
    sampler: EpochRandomXSliceSampler,
    loader: DataLoader[dict[str, Any]],
) -> None:
    print("\nData")
    rows = (
        ("Train LOW directory", DATA.train_low_dir),
        ("Train HIGH directory", DATA.train_high_dir),
        ("Train volumes", dataset.num_volumes),
        ("Full train samples", f"{len(dataset):,}"),
        (
            "X positions/vol/epoch",
            f"{TRAIN.slices_per_volume_per_epoch} / {dataset.samples_per_volume}",
        ),
        ("Samples this epoch", f"{len(sampler):,}"),
        ("Physical batch size", TRAIN.batch_size),
        ("Batches/epoch", f"{len(loader):,}"),
        ("Num workers", TRAIN.num_workers),
        ("Persistent workers", TRAIN.persistent_workers),
        (
            "Prefetch factor",
            TRAIN.prefetch_factor if TRAIN.num_workers > 0 else "N/A",
        ),
    )
    for key, value in rows:
        print(f"  {key:<24}: {value}")


def print_model_summary(model: nn.Module) -> None:
    print("\nModel")
    rows = (
        ("Architecture", "Dual-Path Fully Dense U-Net + SFA + BEFD + AFF (E5)"),
        ("Training plane", "Y-Z"),
        ("Adjacent axis", "X"),
        ("Channels", "3-8-16-32-64-128-64-32-16-8-1"),
        ("Parameters", f"{count_trainable_parameters(model):,}"),
        ("Dense layers/block", MODEL.dense_layers_per_block),
        ("SFA stages", ", ".join(MODEL.sfa_stages)),
        ("SFA dilation rates", MODEL.sfa_dilation_rates),
        ("BEFD boundary stages", ", ".join(MODEL.befd_boundary_stages)),
        ("BEFD denoised skips", ", ".join(MODEL.befd_denoise_skip_stages)),
        ("FD non-local positions", f"<= {MODEL.denoise_max_positions}"),
        ("Boundary path channels", MODEL.boundary_path_channels),
        ("Boundary fusion stages", ", ".join(MODEL.boundary_fusion_stages)),
        ("AFF stages", ", ".join(MODEL.aff_stages)),
        ("AFF context", "Local spatial + global channel"),
        ("BatchNorm", "None"),
        ("Final output activation", "None (Linear)"),
        ("Channels last", TRAIN.use_channels_last),
    )
    for key, value in rows:
        print(f"  {key:<24}: {value}")


def print_training_summary(amp_enabled: bool) -> None:
    print("\nTraining configuration")
    rows = (
        ("Epochs", TRAIN.num_epochs),
        ("Learning rate", TRAIN.learning_rate),
        ("Loss", "0.7 L1 + 0.2 SSIM + 0.1 Sobel Edge"),
        ("SSIM data range", TRAIN.ssim_data_range),
        ("SSIM window/sigma", f"{TRAIN.ssim_window_size} / {TRAIN.ssim_sigma}"),
        ("Optimizer", "Adam"),
        ("AMP", amp_enabled),
        ("Validation", "None"),
        ("Test", "Not used"),
        ("MAP save every", f"{SAVE.map_save_every} epochs"),
        ("MAP volume index", SAVE.map_volume_index),
    )
    for key, value in rows:
        print(f"  {key:<24}: {value}")


# ============================================================
# Full training
# ============================================================


def train() -> None:
    set_seed(TRAIN.seed)
    configure_cuda()
    SAVE.create_directories()

    device = get_device()
    amp_enabled = TRAIN.use_amp and device.type == "cuda"
    print_experiment_header(device)

    train_dataset, train_sampler, train_loader = create_train_loader()
    samples_per_epoch = len(train_sampler)

    try:
        print_data_summary(train_dataset, train_sampler, train_loader)

        model = DualPathFullyDenseSFABEFDAFFUNet().to(device)
        if use_channels_last(device):
            model = model.to(memory_format=torch.channels_last)
        print_model_summary(model)

        criterion = PAMCompositeLoss()
        print("\nOptimizer")
        optimizer = create_optimizer(model, device)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        print_training_summary(amp_enabled)

        train_sampler.set_epoch(0)
        verify_model_shape(model, train_loader, device, amp_enabled)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print_cuda_memory("Before Training")

        if SAVE.save_initial_map:
            print("\nGenerating initial Epoch 0 Train MAP...")
            save_epoch_map(
                model,
                train_dataset,
                device,
                epoch=0,
                amp_enabled=amp_enabled,
                volume_index=SAVE.map_volume_index,
            )

        history: list[HistoryRow] = []
        print("\n" + "=" * 70)
        print("TRAINING START")
        print("=" * 70)

        for epoch in range(1, TRAIN.num_epochs + 1):
            train_sampler.set_epoch(epoch)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
                total_epochs=TRAIN.num_epochs,
                amp_enabled=amp_enabled,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            epoch_time = time.perf_counter() - start_time
            peak_gpu_memory_gb = (
                torch.cuda.max_memory_allocated() / (1024**3)
                if torch.cuda.is_available()
                else 0.0
            )

            history.append(
                HistoryRow(
                    epoch=epoch,
                    train_total_loss=train_metrics.total,
                    train_l1_loss=train_metrics.l1,
                    train_ssim_loss=train_metrics.ssim,
                    train_edge_loss=train_metrics.edge,
                    epoch_time_seconds=epoch_time,
                    peak_gpu_memory_gb=peak_gpu_memory_gb,
                    samples_this_epoch=samples_per_epoch,
                )
            )

            print(f"\nEpoch {epoch:03d}/{TRAIN.num_epochs:03d}")
            print(f"Train Total Loss: {train_metrics.total:.8f}")
            print(f"Train L1 Loss   : {train_metrics.l1:.8f}")
            print(f"Train SSIM Loss : {train_metrics.ssim:.8f}")
            print(f"Train Edge Loss : {train_metrics.edge:.8f}")
            print(f"Epoch Time      : {epoch_time:.2f} sec")
            print(f"Peak GPU Memory : {peak_gpu_memory_gb:.2f} GB")
            print(f"Samples trained : {samples_per_epoch:,}")

            if epoch % SAVE.map_save_every == 0:
                print(f"\nGenerating full-volume Train MAP for Epoch {epoch}...")
                save_epoch_map(
                    model,
                    train_dataset,
                    device,
                    epoch,
                    amp_enabled,
                    SAVE.map_volume_index,
                )

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                train_metrics.total,
                samples_per_epoch,
                SAVE.latest_model_path,
            )

            if epoch % SAVE.checkpoint_every == 0:
                checkpoint_path = SAVE.checkpoint_dir / f"epoch_{epoch:03d}.pth"
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    train_metrics.total,
                    samples_per_epoch,
                    checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")

            save_training_history(history)
            save_loss_curve(history)

        final_train_loss = history[-1].train_total_loss
        save_checkpoint(
            model,
            optimizer,
            scaler,
            TRAIN.num_epochs,
            final_train_loss,
            samples_per_epoch,
            SAVE.final_model_path,
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nFinal model:\n  {SAVE.final_model_path}")
        print(f"\nLatest model:\n  {SAVE.latest_model_path}")
        print(f"\nTraining MAP images:\n  {SAVE.train_map_dir}")
        print(f"\nTraining history:\n  {SAVE.history_csv_path}")
        print(f"\nLoss curve:\n  {SAVE.loss_curve_path}")

    finally:
        train_dataset.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
