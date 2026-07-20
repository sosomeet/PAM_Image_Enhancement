# -*- coding: utf-8 -*-
"""
Pure conditional DDPM for PAM image enhancement.

The original restoration-specific components have been removed:
    - no edge/Sobel/Laplacian branch
    - no Haar-wavelet branch
    - no gated residual fusion
    - no multi-stage refiner
    - no L1/SSIM/boundary/intensity composite objective
    - no adversarial or perceptual objective

The remaining model is a standard conditional denoising diffusion model:

    HIGH target x_0 [B,1,H,W]
        -> choose random diffusion step t
        -> add Gaussian noise epsilon
        -> obtain x_t

    concat(x_t, normalized 3-adjacent LOW condition)
        -> time-conditioned U-Net
        -> predict epsilon

    training loss = MSE(predicted epsilon, true epsilon)

At inference, sampling starts from Gaussian noise and repeatedly applies the
DDPM reverse process while conditioning every denoising step on LOW.

Data specification:
    LOW  : ./data/3d_Train/LOW/*.bin
           float32 + 48-byte header
           [X,C,Y,Z] = [200,3,200,512]

    HIGH : ./data/norm_Train/HIGH/*.bin
           float32 + 48-byte header
           [X,Y,Z] = [200,200,512]

Notes:
    - LOW and HIGH are assumed to be normalized to [0,1].
    - Internally, diffusion operates in [-1,1].
    - This script trains only; no validation/test split is created.
    - Preview images use full ancestral DDPM sampling and therefore are much
      slower than a one-pass restoration network.
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
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42


class DataConfig:
    def __init__(self) -> None:
        self.train_low_dir = Path("./data/3d_Train/LOW")
        self.train_high_dir = Path("./data/norm_Train/HIGH")

        self.header_size = 48
        self.dtype = np.float32

        self.high_volume_shape = (200, 200, 512)      # [X,Y,Z]
        self.adjacent_low_shape = (200, 3, 200, 512)  # [X,C,Y,Z]

        self.condition_channels = 3
        self.target_channels = 1


class ModelConfig:
    def __init__(self) -> None:
        self.condition_channels = 3
        self.target_channels = 1

        # RTX 3060 Ti-oriented width.
        self.base_channels = 32
        self.max_channels = 128
        self.channel_multipliers = (1, 2, 4, 4)
        self.residual_blocks_per_level = 2
        self.time_embedding_dim = 128
        self.dropout = 0.0


class DiffusionConfig:
    def __init__(self) -> None:
        # A 200-step schedule keeps ancestral sampling usable for previews.
        # beta_end is larger than the usual 1000-step value so alpha_bar_T is
        # close enough to zero when using only 200 forward steps.
        self.num_steps = 200
        self.beta_start = 1e-4
        self.beta_end = 5e-2

        # Standard DDPM posterior sampling clips predicted x_0 to the data
        # domain used by diffusion, here [-1,1].
        self.clip_denoised = True


class TrainConfig:
    def __init__(self) -> None:
        self.num_epochs = 50
        self.learning_rate = 2e-4

        # 130 volumes x 32 X positions = 4,160 samples per epoch.
        self.slices_per_volume_per_epoch = 32

        # Diffusion U-Net activation memory is larger than the previous model.
        self.batch_size = 2
        self.gradient_accumulation_steps = 4

        self.num_workers = 4
        self.prefetch_factor = 2
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = True

        self.use_amp = True
        self.use_channels_last = True
        self.grad_clip_norm = 1.0
        self.log_every = 20


class SaveConfig:
    def __init__(self) -> None:
        self.run_root = Path("./14_Pure_Conditional_DDPM")

        self.checkpoint_every = 10
        self.preview_every = 5
        self.preview_volume_index = 0
        self.preview_x_indices = (50, 100, 150)
        self.preview_seed = 1234

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
    def preview_dir(self) -> Path:
        return self.output_root / "sample_previews"

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
DIFFUSION = DiffusionConfig()
TRAIN = TrainConfig()
SAVE = SaveConfig()

X_SIZE, Y_SIZE, Z_SIZE = DATA.high_volume_shape
INPUT_SAMPLE_SHAPE = (DATA.condition_channels, Y_SIZE, Z_SIZE)
TARGET_SAMPLE_SHAPE = (DATA.target_channels, Y_SIZE, Z_SIZE)


# ============================================================
# 2. Reproducibility / runtime setup
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
        SAVE.preview_dir,
        SAVE.history_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. File utilities
# ============================================================


def extract_volume_key(file_path: Path) -> str:
    """
    Train_000_LOW.bin  -> Train_000
    Train_000_HiGH.bin -> Train_000
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
# 4. Dataset and sampler
# ============================================================


class PAMAdjacentYZDataset(Dataset):
    """
    LOW input is pre-generated 3-adjacent Y-Z data.

    One sample at x_index:
        condition = LOW[x_index]  -> [3,200,512]
        target    = HIGH[x_index] -> [1,200,512]
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

        condition_array = np.array(
            low_volume[x_index],
            dtype=np.float32,
            copy=True,
        )
        target_array = np.array(
            high_volume[x_index],
            dtype=np.float32,
            copy=True,
        )[None, ...]

        if condition_array.shape != INPUT_SAMPLE_SHAPE:
            raise ValueError(
                "\nCondition shape mismatch\n"
                f"Volume key    : {pair['volume_key']}\n"
                f"X index       : {x_index}\n"
                f"Actual shape  : {condition_array.shape}\n"
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
            "input": torch.from_numpy(condition_array),
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
    """Sample unique X positions from every volume at each epoch."""

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
# 5. Pure time-conditioned U-Net denoiser
# ============================================================


def group_count(channels: int, maximum_groups: int = 8) -> int:
    groups = min(maximum_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal diffusion-step embedding."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        if embedding_dim < 4:
            raise ValueError("embedding_dim must be >= 4")

        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape [B], got {tuple(timesteps.shape)}"
            )

        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half_dim - 1, 1)

        frequencies = torch.exp(exponent)
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)

        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class DiffusionResidualBlock(nn.Module):
    """
    Standard residual block with additive timestep conditioning.

    No edge, wavelet, attention, dense, AFF, SFA, or refinement branch is used.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(x)

        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.time_projection(time_embedding)[:, :, None, None]
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))

        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_blocks: int,
        dropout: float,
        add_downsample: bool,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        current_channels = in_channels

        for _ in range(num_blocks):
            blocks.append(
                DiffusionResidualBlock(
                    current_channels,
                    out_channels,
                    time_embedding_dim,
                    dropout,
                )
            )
            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.downsample = Downsample(out_channels) if add_downsample else None

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, time_embedding)

        skip = x

        if self.downsample is not None:
            x = self.downsample(x)

        return x, skip


class UpLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        num_blocks: int,
        dropout: float,
        add_upsample: bool,
    ) -> None:
        super().__init__()

        self.upsample = Upsample(in_channels) if add_upsample else None

        blocks: list[nn.Module] = []
        current_channels = in_channels + skip_channels

        for block_index in range(num_blocks):
            block_out_channels = out_channels
            blocks.append(
                DiffusionResidualBlock(
                    current_channels,
                    block_out_channels,
                    time_embedding_dim,
                    dropout,
                )
            )
            current_channels = block_out_channels

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)

        for block in self.blocks:
            x = block(x, time_embedding)

        return x


class ConditionalDiffusionUNet(nn.Module):
    """
    Vanilla time-conditioned U-Net noise predictor.

    Inputs:
        noisy_target : [B,1,H,W] = x_t
        condition    : [B,3,H,W] = normalized 3-adjacent LOW
        timesteps    : [B]

    Output:
        predicted Gaussian noise epsilon_theta(x_t, condition, t)
    """

    def __init__(
        self,
        condition_channels: int = 3,
        target_channels: int = 1,
        base_channels: int = 32,
        max_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        residual_blocks_per_level: int = 2,
        time_embedding_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least 2 levels")

        self.condition_channels = int(condition_channels)
        self.target_channels = int(target_channels)
        self.num_levels = len(channel_multipliers)
        self.required_multiple = 2 ** (self.num_levels - 1)

        channels = tuple(
            min(base_channels * multiplier, max_channels)
            for multiplier in channel_multipliers
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.input_conv = nn.Conv2d(
            target_channels + condition_channels,
            channels[0],
            kernel_size=3,
            padding=1,
        )

        down_levels: list[nn.Module] = []
        current_channels = channels[0]

        for level_index, level_channels in enumerate(channels):
            down_levels.append(
                DownLevel(
                    in_channels=current_channels,
                    out_channels=level_channels,
                    time_embedding_dim=time_embedding_dim,
                    num_blocks=residual_blocks_per_level,
                    dropout=dropout,
                    add_downsample=level_index < self.num_levels - 1,
                )
            )
            current_channels = level_channels

        self.down_levels = nn.ModuleList(down_levels)

        self.middle_block1 = DiffusionResidualBlock(
            channels[-1],
            channels[-1],
            time_embedding_dim,
            dropout,
        )
        self.middle_block2 = DiffusionResidualBlock(
            channels[-1],
            channels[-1],
            time_embedding_dim,
            dropout,
        )

        up_levels: list[nn.Module] = []
        current_channels = channels[-1]

        for reverse_index, level_index in enumerate(
            reversed(range(self.num_levels))
        ):
            skip_channels = channels[level_index]
            out_channels = channels[level_index]

            up_levels.append(
                UpLevel(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    time_embedding_dim=time_embedding_dim,
                    num_blocks=residual_blocks_per_level,
                    dropout=dropout,
                    add_upsample=reverse_index > 0,
                )
            )
            current_channels = out_channels

        self.up_levels = nn.ModuleList(up_levels)

        self.output_norm = nn.GroupNorm(
            group_count(channels[0]),
            channels[0],
        )
        self.output_conv = nn.Conv2d(
            channels[0],
            target_channels,
            kernel_size=3,
            padding=1,
        )

        # A zero-initialized output layer is standard and stabilizes the start
        # of epsilon-prediction training.
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor,
        multiple: int,
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

        h_end = x.shape[-2] - pad_bottom if pad_bottom else x.shape[-2]
        w_end = x.shape[-1] - pad_right if pad_right else x.shape[-1]

        return x[..., pad_top:h_end, pad_left:w_end]

    def forward(
        self,
        noisy_target: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.ndim != 4:
            raise ValueError(
                f"noisy_target must be [B,C,H,W], got {tuple(noisy_target.shape)}"
            )

        if condition.ndim != 4:
            raise ValueError(
                f"condition must be [B,C,H,W], got {tuple(condition.shape)}"
            )

        if noisy_target.size(1) != self.target_channels:
            raise ValueError(
                f"Expected {self.target_channels} target channels, "
                f"got {noisy_target.size(1)}"
            )

        if condition.size(1) != self.condition_channels:
            raise ValueError(
                f"Expected {self.condition_channels} condition channels, "
                f"got {condition.size(1)}"
            )

        if noisy_target.shape[0] != condition.shape[0]:
            raise ValueError("Batch size mismatch between target and condition")

        if noisy_target.shape[-2:] != condition.shape[-2:]:
            raise ValueError("Spatial size mismatch between target and condition")

        if timesteps.shape != (noisy_target.shape[0],):
            raise ValueError(
                f"timesteps must have shape {(noisy_target.shape[0],)}, "
                f"got {tuple(timesteps.shape)}"
            )

        original_size = noisy_target.shape[-2:]

        noisy_target, padding = self._pad_to_multiple(
            noisy_target,
            self.required_multiple,
        )
        condition, condition_padding = self._pad_to_multiple(
            condition,
            self.required_multiple,
        )

        if condition_padding != padding:
            raise RuntimeError("Target and condition padding unexpectedly differ")

        time_embedding = self.time_mlp(timesteps)
        x = self.input_conv(torch.cat([noisy_target, condition], dim=1))

        skips: list[torch.Tensor] = []
        for level in self.down_levels:
            x, skip = level(x, time_embedding)
            skips.append(skip)

        x = self.middle_block1(x, time_embedding)
        x = self.middle_block2(x, time_embedding)

        for level, skip in zip(self.up_levels, reversed(skips)):
            x = level(x, skip, time_embedding)

        x = self.output_conv(F.silu(self.output_norm(x)))
        x = self._remove_padding(x, padding)

        if x.shape[-2:] != original_size:
            raise RuntimeError(
                f"Denoiser output shape mismatch: {tuple(x.shape[-2:])} "
                f"!= {original_size}"
            )

        return x


# ============================================================
# 6. Pure conditional DDPM
# ============================================================


def extract_schedule_values(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Gather one scalar schedule value per batch item and reshape to BCHW."""
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], 1, 1, 1).to(
        device=reference.device,
        dtype=reference.dtype,
    )


class ConditionalDDPM(nn.Module):
    """
    Conditional denoising diffusion probabilistic model.

    Default forward behavior performs full ancestral sampling so existing
    evaluators can call `prediction = model(low_condition)`.

    Training uses `model.training_loss(condition, target)`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        max_channels: int = 128,
        diffusion_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 5e-2,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 4),
        residual_blocks_per_level: int = 2,
        time_embedding_dim: int = 128,
        dropout: float = 0.0,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()

        if diffusion_steps <= 1:
            raise ValueError("diffusion_steps must be > 1")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError(
                "Require 0 < beta_start < beta_end < 1, got "
                f"{beta_start}, {beta_end}"
            )

        self.condition_channels = int(in_channels)
        self.target_channels = int(out_channels)
        self.num_steps = int(diffusion_steps)
        self.clip_denoised = bool(clip_denoised)

        self.denoiser = ConditionalDiffusionUNet(
            condition_channels=in_channels,
            target_channels=out_channels,
            base_channels=base_channels,
            max_channels=max_channels,
            channel_multipliers=channel_multipliers,
            residual_blocks_per_level=residual_blocks_per_level,
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
        )

        betas = torch.linspace(
            beta_start,
            beta_end,
            diffusion_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(
            alpha_cumprod[:-1],
            (1, 0),
            value=1.0,
        )

        posterior_variance = (
            betas
            * (1.0 - alpha_cumprod_prev)
            / (1.0 - alpha_cumprod)
        )

        posterior_mean_coef1 = (
            betas
            * torch.sqrt(alpha_cumprod_prev)
            / (1.0 - alpha_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alpha_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alpha_cumprod)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer(
            "sqrt_alpha_cumprod",
            torch.sqrt(alpha_cumprod),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alpha_cumprod",
            torch.sqrt(1.0 / alpha_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alpha_cumprod",
            torch.sqrt(1.0 / alpha_cumprod - 1.0),
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @staticmethod
    def normalize_to_diffusion_range(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

    @staticmethod
    def unnormalize_from_diffusion_range(x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = extract_schedule_values(
            self.sqrt_alpha_cumprod,
            timesteps,
            x_start,
        )
        sqrt_one_minus_alpha_bar = extract_schedule_values(
            self.sqrt_one_minus_alpha_cumprod,
            timesteps,
            x_start,
        )

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_x_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        reciprocal = extract_schedule_values(
            self.sqrt_recip_alpha_cumprod,
            timesteps,
            x_t,
        )
        reciprocal_minus_one = extract_schedule_values(
            self.sqrt_recipm1_alpha_cumprod,
            timesteps,
            x_t,
        )

        return reciprocal * x_t - reciprocal_minus_one * predicted_noise

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            extract_schedule_values(
                self.posterior_mean_coef1,
                timesteps,
                x_t,
            )
            * x_start
            + extract_schedule_values(
                self.posterior_mean_coef2,
                timesteps,
                x_t,
            )
            * x_t
        )

        posterior_variance = extract_schedule_values(
            self.posterior_variance,
            timesteps,
            x_t,
        )
        posterior_log_variance = extract_schedule_values(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t,
        )

        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        normalized_condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = self.denoiser(
            noisy_target=x_t,
            condition=normalized_condition,
            timesteps=timesteps,
        )

        predicted_x_start = self.predict_x_start_from_noise(
            x_t=x_t,
            timesteps=timesteps,
            predicted_noise=predicted_noise,
        )

        if self.clip_denoised:
            predicted_x_start = predicted_x_start.clamp(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior(
                x_start=predicted_x_start,
                x_t=x_t,
                timesteps=timesteps,
            )
        )

        return (
            model_mean,
            posterior_variance,
            posterior_log_variance,
            predicted_x_start,
        )

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        normalized_condition: torch.Tensor,
        timesteps: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_mean, _, model_log_variance, predicted_x_start = (
            self.p_mean_variance(
                x_t=x_t,
                normalized_condition=normalized_condition,
                timesteps=timesteps,
            )
        )

        noise = torch.randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        nonzero_mask = (timesteps != 0).to(x_t.dtype).reshape(
            x_t.shape[0], 1, 1, 1
        )

        sample = (
            model_mean
            + nonzero_mask
            * torch.exp(0.5 * model_log_variance)
            * noise
        )

        return sample, predicted_x_start

    def training_loss(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Standard epsilon-prediction objective from DDPM."""
        if condition.ndim != 4 or target.ndim != 4:
            raise ValueError("condition and target must both be BCHW tensors")

        batch_size = target.shape[0]
        timesteps = torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            device=target.device,
            dtype=torch.long,
        )

        normalized_condition = self.normalize_to_diffusion_range(condition)
        x_start = self.normalize_to_diffusion_range(target)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, timesteps, noise)

        predicted_noise = self.denoiser(
            noisy_target=x_t,
            condition=normalized_condition,
            timesteps=timesteps,
        )

        noise_mse = F.mse_loss(predicted_noise, noise)

        # Reconstruction L1 is logged only. It is not added to the objective.
        with torch.no_grad():
            predicted_x_start = self.predict_x_start_from_noise(
                x_t=x_t,
                timesteps=timesteps,
                predicted_noise=predicted_noise,
            )
            predicted_target = self.unnormalize_from_diffusion_range(
                predicted_x_start.clamp(-1.0, 1.0)
            )
            x0_l1 = F.l1_loss(predicted_target, target)

        return noise_mse, {
            "noise_mse": noise_mse.detach(),
            "x0_l1": x0_l1.detach(),
            "mean_timestep": timesteps.float().mean().detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        generator: torch.Generator | None = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Full ancestral DDPM sampling conditioned on LOW."""
        if condition.ndim != 4:
            raise ValueError(
                f"condition must be [B,C,H,W], got {tuple(condition.shape)}"
            )

        if condition.size(1) != self.condition_channels:
            raise ValueError(
                f"Expected {self.condition_channels} condition channels, "
                f"got {condition.size(1)}"
            )

        normalized_condition = self.normalize_to_diffusion_range(condition)

        x_t = torch.randn(
            (
                condition.shape[0],
                self.target_channels,
                condition.shape[-2],
                condition.shape[-1],
            ),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )

        iterator = reversed(range(self.num_steps))
        if show_progress:
            iterator = tqdm(
                iterator,
                total=self.num_steps,
                desc="DDPM sampling",
                leave=False,
                dynamic_ncols=True,
            )

        predicted_x_start = x_t

        for step in iterator:
            timesteps = torch.full(
                (condition.shape[0],),
                step,
                device=condition.device,
                dtype=torch.long,
            )
            x_t, predicted_x_start = self.p_sample(
                x_t=x_t,
                normalized_condition=normalized_condition,
                timesteps=timesteps,
                generator=generator,
            )

        prediction = self.unnormalize_from_diffusion_range(
            predicted_x_start
        )
        return prediction.clamp(0.0, 1.0)

    def forward(
        self,
        condition: torch.Tensor,
        noisy_target: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Evaluator-compatible behavior:
            model(condition) -> full DDPM sample

        Low-level denoiser behavior:
            model(condition, noisy_target, timesteps) -> predicted noise
        """
        if noisy_target is None and timesteps is None:
            return self.sample(condition)

        if noisy_target is None or timesteps is None:
            raise ValueError(
                "noisy_target and timesteps must be provided together"
            )

        normalized_condition = self.normalize_to_diffusion_range(condition)
        return self.denoiser(
            noisy_target=noisy_target,
            condition=normalized_condition,
            timesteps=timesteps,
        )


# Evaluator-friendly aliases.
PAMEnhancer = ConditionalDDPM
UNet = ConditionalDDPM
Model = ConditionalDDPM


# ============================================================
# 7. History and visualization
# ============================================================


class HistoryRow:
    def __init__(
        self,
        epoch: int,
        total_loss: float,
        noise_mse: float,
        x0_l1: float,
        mean_timestep: float,
        learning_rate: float,
        epoch_seconds: float,
    ) -> None:
        self.epoch = int(epoch)
        self.total_loss = float(total_loss)
        self.noise_mse = float(noise_mse)
        self.x0_l1 = float(x0_l1)
        self.mean_timestep = float(mean_timestep)
        self.learning_rate = float(learning_rate)
        self.epoch_seconds = float(epoch_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "noise_mse": self.noise_mse,
            "x0_l1": self.x0_l1,
            "mean_timestep": self.mean_timestep,
            "learning_rate": self.learning_rate,
            "epoch_seconds": self.epoch_seconds,
        }



def load_preview_slices(
    dataset: PAMAdjacentYZDataset,
    volume_index: int,
    x_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, str]:
    if volume_index < 0 or volume_index >= dataset.num_volumes:
        raise IndexError(
            f"volume_index={volume_index} outside "
            f"0~{dataset.num_volumes - 1}"
        )

    for x_index in x_indices:
        if x_index < 0 or x_index >= X_SIZE:
            raise IndexError(f"Invalid preview x_index: {x_index}")

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

    conditions = np.array(
        low_volume[list(x_indices)],
        dtype=np.float32,
        copy=True,
    )
    targets = np.array(
        high_volume[list(x_indices)],
        dtype=np.float32,
        copy=True,
    )[:, None, :, :]

    del low_volume
    del high_volume

    return conditions, targets, str(pair["volume_key"])



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



def make_device_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator



def save_sampling_preview(
    model: ConditionalDDPM,
    dataset: PAMAdjacentYZDataset,
    device: torch.device,
    epoch: int,
) -> None:
    model_was_training = model.training
    model.eval()

    conditions_np, targets_np, volume_key = load_preview_slices(
        dataset=dataset,
        volume_index=SAVE.preview_volume_index,
        x_indices=SAVE.preview_x_indices,
    )

    conditions = torch.from_numpy(conditions_np).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    if TRAIN.use_channels_last and device.type == "cuda":
        conditions = conditions.contiguous(memory_format=torch.channels_last)

    generator = make_device_generator(
        device=device,
        seed=SAVE.preview_seed,
    )
    amp_enabled = TRAIN.use_amp and device.type == "cuda"

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model.sample(
                condition=conditions,
                generator=generator,
                show_progress=True,
            )

    predictions_np = predictions.float().cpu().numpy()

    num_rows = len(SAVE.preview_x_indices)
    fig, axes = plt.subplots(num_rows, 4, figsize=(22, 5.5 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, x_index in enumerate(SAVE.preview_x_indices):
        low_image = np.rot90(conditions_np[row_index, 1], k=-1)
        prediction = np.rot90(predictions_np[row_index, 0], k=-1)
        target = np.rot90(targets_np[row_index, 0], k=-1)
        error = np.abs(target - prediction)

        v_min, v_max = robust_limits([low_image, prediction, target])
        error_max = float(np.percentile(error, 99.8))
        if error_max <= 0:
            error_max = 1e-8

        panels = (
            (low_image, "LOW center", "hot", v_min, v_max),
            (prediction, "DDPM sample", "hot", v_min, v_max),
            (target, "HIGH target", "hot", v_min, v_max),
            (error, "Absolute error", "viridis", 0.0, error_max),
        )

        for axis, (image, title, cmap, p_min, p_max) in zip(
            axes[row_index], panels
        ):
            rendered = axis.imshow(
                image,
                cmap=cmap,
                origin="upper",
                aspect="auto",
                vmin=p_min,
                vmax=p_max,
            )
            axis.set_title(f"{title} | X={x_index}")
            axis.set_xlabel("Y")
            axis.set_ylabel("Z")
            fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Pure Conditional DDPM | {volume_key} | Epoch {epoch:03d} | "
        f"{model.num_steps} ancestral steps | 90° clockwise"
    )
    fig.tight_layout()

    save_path = SAVE.preview_dir / f"epoch_{epoch:03d}_samples.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    del conditions
    del predictions

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if model_was_training:
        model.train()


# ============================================================
# 8. Checkpoint and history utilities
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
        "architecture": "ConditionalDDPM",
        "denoiser": "time-conditioned U-Net without attention or auxiliary branches",
        "conditioning": "concatenate x_t with normalized 3-adjacent LOW",
        "prediction_target": "Gaussian noise epsilon",
        "training_loss": "MSE(predicted_noise, true_noise)",
        "input_shape": list(INPUT_SAMPLE_SHAPE),
        "target_shape": list(TARGET_SAMPLE_SHAPE),
        "data_low_shape": list(DATA.adjacent_low_shape),
        "data_high_shape": list(DATA.high_volume_shape),
        "data_range_external": "[0,1]",
        "diffusion_range_internal": "[-1,1]",
        "dtype": str(np.dtype(DATA.dtype)),
        "header_size": DATA.header_size,
        "center_channel": 1,
        "base_channels": MODEL.base_channels,
        "max_channels": MODEL.max_channels,
        "channel_multipliers": list(MODEL.channel_multipliers),
        "residual_blocks_per_level": MODEL.residual_blocks_per_level,
        "time_embedding_dim": MODEL.time_embedding_dim,
        "dropout": MODEL.dropout,
        "diffusion_steps": DIFFUSION.num_steps,
        "beta_schedule": "linear",
        "beta_start": DIFFUSION.beta_start,
        "beta_end": DIFFUSION.beta_end,
        "clip_denoised": DIFFUSION.clip_denoised,
        "sampler": "ancestral DDPM",
        "optimizer": "Adam",
        "learning_rate": TRAIN.learning_rate,
        "weight_decay": 0.0,
        "physical_batch_size": TRAIN.batch_size,
        "gradient_accumulation_steps": TRAIN.gradient_accumulation_steps,
        "effective_batch_size": (
            TRAIN.batch_size * TRAIN.gradient_accumulation_steps
        ),
        "slices_per_volume_per_epoch": TRAIN.slices_per_volume_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "use_amp": TRAIN.use_amp,
        "use_channels_last": TRAIN.use_channels_last,
    }



def save_checkpoint(
    model: ConditionalDDPM,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
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
    if not history:
        return

    fieldnames = list(history[0].to_dict().keys())

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
        [row.noise_mse for row in history],
        marker="o",
        markersize=3,
        label="Noise MSE",
    )
    plt.plot(
        [row.epoch for row in history],
        [row.x0_l1 for row in history],
        marker="o",
        markersize=3,
        label="Logged x0 L1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Pure Conditional DDPM Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE.loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 9. Compatibility helpers
# ============================================================



def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ============================================================
# 10. Smoke test
# ============================================================



def run_model_smoke_test(device: torch.device) -> None:
    """Test noise prediction, loss, backward, and short ancestral sampling."""
    model = ConditionalDDPM(
        in_channels=MODEL.condition_channels,
        out_channels=MODEL.target_channels,
        base_channels=16,
        max_channels=64,
        diffusion_steps=4,
        beta_start=1e-4,
        beta_end=2e-2,
        channel_multipliers=(1, 2, 4),
        residual_blocks_per_level=1,
        time_embedding_dim=64,
        dropout=0.0,
    ).to(device)

    condition = torch.rand(1, 3, 32, 64, device=device)
    target = torch.rand(1, 1, 32, 64, device=device)

    loss, metrics = model.training_loss(condition, target)

    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("Smoke test failed: invalid diffusion loss")

    loss.backward()

    model.eval()
    generator = make_device_generator(device, seed=SEED)
    with torch.inference_mode():
        sample = model.sample(condition, generator=generator)

    if sample.shape != target.shape:
        raise RuntimeError(
            f"Smoke test failed: sample shape={tuple(sample.shape)}"
        )

    if not torch.isfinite(sample).all():
        raise RuntimeError("Smoke test failed: sample contains NaN/Inf")

    if not {"noise_mse", "x0_l1", "mean_timestep"}.issubset(metrics):
        raise RuntimeError("Smoke test failed: missing training metrics")

    del model
    del condition
    del target
    del loss
    del sample

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 11. Training
# ============================================================



def train() -> None:
    set_seed(SEED)
    configure_cuda()
    create_output_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 92)
    print("PAM IMAGE ENHANCEMENT — PURE CONDITIONAL DDPM")
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
    print(f"Channel multipliers       : {MODEL.channel_multipliers}")
    print(f"Diffusion steps           : {DIFFUSION.num_steps}")
    print(
        f"Linear beta schedule      : "
        f"{DIFFUSION.beta_start:g} -> {DIFFUSION.beta_end:g}"
    )
    print("Conditioning              : concat(x_t, 3-adjacent LOW)")
    print("Prediction target         : Gaussian noise epsilon")
    print("Training loss             : noise MSE only")
    print("Sampling                  : ancestral DDPM")
    print("=" * 92)

    print("\n[1/5] Running diffusion smoke test...")
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

    print("\n[3/5] Building pure conditional DDPM...")
    model = ConditionalDDPM(
        in_channels=MODEL.condition_channels,
        out_channels=MODEL.target_channels,
        base_channels=MODEL.base_channels,
        max_channels=MODEL.max_channels,
        diffusion_steps=DIFFUSION.num_steps,
        beta_start=DIFFUSION.beta_start,
        beta_end=DIFFUSION.beta_end,
        channel_multipliers=MODEL.channel_multipliers,
        residual_blocks_per_level=MODEL.residual_blocks_per_level,
        time_embedding_dim=MODEL.time_embedding_dim,
        dropout=MODEL.dropout,
        clip_denoised=DIFFUSION.clip_denoised,
    ).to(device)

    if TRAIN.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_params, trainable_params = count_parameters(model)
    print(f"      Total parameters    : {total_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")
    print(
        f"      alpha_bar(T-1)      : "
        f"{float(model.alpha_cumprod[-1].item()):.8f}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN.learning_rate,
        betas=(0.9, 0.999),
    )

    amp_enabled = TRAIN.use_amp and device.type == "cuda"
    scaler = create_grad_scaler(enabled=amp_enabled)

    print("\n[4/5] Starting training...")

    history: list[HistoryRow] = []

    try:
        for epoch in range(1, TRAIN.num_epochs + 1):
            epoch_start = time.time()
            sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_total = 0.0
            running_noise_mse = 0.0
            running_x0_l1 = 0.0
            running_mean_timestep = 0.0
            seen_batches = 0
            accumulation_counter = 0

            progress = tqdm(
                loader,
                desc=f"Epoch {epoch:03d}/{TRAIN.num_epochs:03d}",
                dynamic_ncols=True,
            )

            for step, batch in enumerate(progress, start=1):
                conditions = batch["input"].to(
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
                    conditions = conditions.contiguous(
                        memory_format=torch.channels_last
                    )
                    targets = targets.contiguous(
                        memory_format=torch.channels_last
                    )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    total_loss, metrics = model.training_loss(
                        condition=conditions,
                        target=targets,
                    )
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
                running_noise_mse += float(metrics["noise_mse"].item())
                running_x0_l1 += float(metrics["x0_l1"].item())
                running_mean_timestep += float(
                    metrics["mean_timestep"].item()
                )
                seen_batches += 1

                if step % TRAIN.log_every == 0 or step == len(loader):
                    progress.set_postfix(
                        MSE=f"{running_noise_mse / seen_batches:.6f}",
                        x0_L1=f"{running_x0_l1 / seen_batches:.5f}",
                        t=f"{running_mean_timestep / seen_batches:.1f}",
                    )

            epoch_seconds = time.time() - epoch_start
            denominator = max(seen_batches, 1)

            row = HistoryRow(
                epoch=epoch,
                total_loss=running_total / denominator,
                noise_mse=running_noise_mse / denominator,
                x0_l1=running_x0_l1 / denominator,
                mean_timestep=running_mean_timestep / denominator,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                epoch_seconds=epoch_seconds,
            )
            history.append(row)

            print(
                f"Epoch {epoch:03d} finished | "
                f"Noise MSE={row.noise_mse:.8f} | "
                f"Logged x0 L1={row.x0_l1:.8f} | "
                f"Mean t={row.mean_timestep:.2f} | "
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
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    history_row=row,
                    samples_per_epoch=samples_per_epoch,
                    path=SAVE.checkpoint_dir / f"epoch_{epoch:03d}.pth",
                )

            save_training_history(history)
            save_loss_curve(history)

            if epoch % SAVE.preview_every == 0:
                print(
                    f"      Sampling preview for epoch {epoch:03d} "
                    f"({DIFFUSION.num_steps} reverse steps)..."
                )
                save_sampling_preview(
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
        print(f"Previews    : {SAVE.preview_dir}")
        print(f"History     : {SAVE.history_csv_path}")
        print("=" * 92)

    finally:
        dataset.close()


# ============================================================
# 12. Entry point
# ============================================================


if __name__ == "__main__":
    train()
