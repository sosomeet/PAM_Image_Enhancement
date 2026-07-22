# -*- coding: utf-8 -*-
"""
Pure Vanilla U-Net for PAM image enhancement.

This script keeps the same training-artifact structure as the conditional DDPM
script, but the network itself is a plain 2D U-Net:

    3-adjacent LOW Y-Z slice [B,3,200,512]
        -> Vanilla U-Net
        -> HIGH Y-Z prediction [B,1,200,512]

Included architecture components
--------------------------------
- Double 3x3 convolution blocks
- 2x2 max pooling
- Transposed-convolution upsampling
- Encoder-to-decoder skip concatenation
- Linear 1x1 output layer

Excluded architecture components
--------------------------------
- Attention gates
- Residual blocks
- Dense blocks
- BEFD / SFA / AFF
- Edge or wavelet branches
- GAN discriminator
- Diffusion process
- Multi-stage refinement

Training objective
------------------
- L1 reconstruction loss only
- MSE, PSNR, and SSIM are logged as evaluation metrics

Data specification
------------------
LOW  : ./data/3d_Train/LOW/*.bin
       float32 + 48-byte header
       [X,C,Y,Z] = [200,3,200,512]

HIGH : ./data/norm_Train/HIGH/*.bin
       float32 + 48-byte header
       [X,Y,Z] = [200,200,512]

Generated artifacts
-------------------
05_Vanilla_U-Net/
├─ models_enhancement/
│  ├─ latest/model_latest.pth
│  ├─ checkpoints/epoch_010.pth, ...
│  └─ final/model_final.pth
└─ outputs/
   ├─ sample_previews/epoch_005_samples.png, ...
   └─ history/
      ├─ training_history.csv
      └─ training_loss_curve.png
"""

from __future__ import annotations

import csv
import math
import random
import re
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DataConfig:
    train_low_dir: Path = Path("./data/3d_Train/LOW")
    train_high_dir: Path = Path("./data/norm_Train/HIGH")

    header_size: int = 48
    dtype: type[np.float32] = np.float32

    high_volume_shape: tuple[int, int, int] = (200, 200, 512)      # [X,Y,Z]
    adjacent_low_shape: tuple[int, int, int, int] = (200, 3, 200, 512)  # [X,C,Y,Z]

    input_channels: int = 3
    output_channels: int = 1


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 3
    output_channels: int = 1

    # Plain U-Net channels: 32 -> 64 -> 128 -> 256 -> 256.
    # The bottleneck is capped at 256 for RTX 3060 Ti-class memory.
    base_channels: int = 32
    max_channels: int = 256
    use_batch_norm: bool = True


@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # 130 volumes x 32 X positions = 4,160 samples/epoch.
    slices_per_volume_per_epoch: int = 32

    batch_size: int = 4
    gradient_accumulation_steps: int = 1

    num_workers: int = 0
    prefetch_factor: int = 2
    use_amp: bool = True
    use_channels_last: bool = True
    grad_clip_norm: float = 1.0
    log_every: int = 20

    @property
    def pin_memory(self) -> bool:
        return torch.cuda.is_available()

    @property
    def persistent_workers(self) -> bool:
        return self.num_workers > 0


@dataclass(frozen=True)
class SaveConfig:
    run_root: Path = Path("./05_Vanilla_U-Net")

    checkpoint_every: int = 10
    preview_every: int = 5
    preview_volume_index: int = 0
    preview_x_indices: tuple[int, ...] = (50, 100, 150)

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
TRAIN = TrainConfig()
SAVE = SaveConfig()

X_SIZE, Y_SIZE, Z_SIZE = DATA.high_volume_shape
INPUT_SAMPLE_SHAPE = (DATA.input_channels, Y_SIZE, Z_SIZE)
TARGET_SAMPLE_SHAPE = (DATA.output_channels, Y_SIZE, Z_SIZE)


# ============================================================
# 2. Runtime setup
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

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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
    """Train_000_LOW.bin / Train_000_HiGH.bin -> Train_000."""
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


def build_file_pairs(low_dir: Path, high_dir: Path) -> list[dict[str, Any]]:
    low_map = {
        extract_volume_key(path): path
        for path in get_bin_files(low_dir)
    }
    high_map = {
        extract_volume_key(path): path
        for path in get_bin_files(high_dir)
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
        input  = LOW[x_index]  -> [3,200,512]
        target = HIGH[x_index] -> [1,200,512]
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
            self._low_memmaps[volume_index] = np.memmap(
                filename=self.pairs[volume_index]["low_path"],
                dtype=DATA.dtype,
                mode="r",
                offset=DATA.header_size,
                shape=DATA.adjacent_low_shape,
                order="C",
            )
        return self._low_memmaps[volume_index]

    def _get_high_memmap(self, volume_index: int) -> np.memmap:
        if volume_index not in self._high_memmaps:
            self._high_memmaps[volume_index] = np.memmap(
                filename=self.pairs[volume_index]["high_path"],
                dtype=DATA.dtype,
                mode="r",
                offset=DATA.header_size,
                shape=DATA.high_volume_shape,
                order="C",
            )
        return self._high_memmaps[volume_index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(
                f"Dataset index out of range: {index}; "
                f"valid range is 0~{len(self) - 1}."
            )

        volume_index, x_index = divmod(index, self.samples_per_volume)
        pair = self.pairs[volume_index]

        low_volume = self._get_low_memmap(volume_index)
        high_volume = self._get_high_memmap(volume_index)

        input_array = np.array(
            low_volume[x_index],
            dtype=np.float32,
            copy=True,
        )
        target_array = np.array(
            high_volume[x_index],
            dtype=np.float32,
            copy=True,
        )[None, ...]

        if input_array.shape != INPUT_SAMPLE_SHAPE:
            raise ValueError(
                f"Input shape mismatch for {pair['volume_key']} x={x_index}: "
                f"{input_array.shape} != {INPUT_SAMPLE_SHAPE}"
            )

        if target_array.shape != TARGET_SAMPLE_SHAPE:
            raise ValueError(
                f"Target shape mismatch for {pair['volume_key']} x={x_index}: "
                f"{target_array.shape} != {TARGET_SAMPLE_SHAPE}"
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
    """Sample a fixed number of unique X positions per volume each epoch."""

    def __init__(
        self,
        dataset: PAMAdjacentYZDataset,
        slices_per_volume: int,
        seed: int = 42,
    ) -> None:
        if not 0 < slices_per_volume <= dataset.samples_per_volume:
            raise ValueError(
                f"slices_per_volume must be in [1, "
                f"{dataset.samples_per_volume}], got {slices_per_volume}."
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
# 5. Pure Vanilla U-Net
# ============================================================


class DoubleConv(nn.Module):
    """(3x3 Conv -> BN -> ReLU) x2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not MODEL.use_batch_norm,
            )
        ]

        if MODEL.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not MODEL.use_batch_norm,
            )
        )

        if MODEL.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """2x up-convolution, skip concatenation, then DoubleConv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

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

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class VanillaUNet(nn.Module):
    """
    Plain four-level 2D U-Net for PAM enhancement.

    Input : [B,3,H,W]
    Output: [B,1,H,W]

    No attention, residual, dense, edge, wavelet, AFF, SFA, BEFD,
    adversarial, diffusion, or multi-stage module is used.
    """

    def __init__(
        self,
        in_channels: int = MODEL.input_channels,
        out_channels: int = MODEL.output_channels,
        base_channels: int = MODEL.base_channels,
        max_channels: int = MODEL.max_channels,
    ) -> None:
        super().__init__()

        if in_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {in_channels}.")
        if out_channels != 1:
            raise ValueError(f"Expected 1 output channel, got {out_channels}.")

        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)
        c4 = min(base_channels * 8, max_channels)
        c5 = min(base_channels * 16, max_channels)

        self.required_multiple = 16
        self.channels = (c1, c2, c3, c4, c5)

        self.encoder1 = DoubleConv(in_channels, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder4 = UpBlock(c5, c4, c4)
        self.decoder3 = UpBlock(c4, c3, c3)
        self.decoder2 = UpBlock(c3, c2, c2)
        self.decoder1 = UpBlock(c2, c1, c1)

        # Linear regression output, consistent with the other enhancement models.
        self.output_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"VanillaUNet expects [B,3,H,W], got {tuple(x.shape)}."
            )

        original_size = x.shape[-2:]
        x, padding = self._pad_to_multiple(x, self.required_multiple)

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(b, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)

        output = self.output_conv(d1)
        output = self._remove_padding(output, padding)

        if output.shape[-2:] != original_size:
            raise RuntimeError(
                f"Output shape mismatch: {tuple(output.shape[-2:])} "
                f"!= {tuple(original_size)}"
            )

        return output


# Evaluator-friendly aliases.
PAMEnhancer = VanillaUNet
UNet = VanillaUNet
Model = VanillaUNet


# ============================================================
# 6. Metrics
# ============================================================


def gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.arange(
        window_size,
        dtype=dtype,
        device=device,
    )
    coordinates -= window_size // 2

    gaussian = torch.exp(
        -(coordinates ** 2) / (2.0 * sigma ** 2)
    )
    gaussian /= gaussian.sum()

    kernel_2d = gaussian[:, None] @ gaussian[None, :]
    return kernel_2d.expand(
        channels,
        1,
        window_size,
        window_size,
    ).contiguous()


def calculate_ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM shape mismatch: {tuple(prediction.shape)} "
            f"vs {tuple(target.shape)}"
        )

    channels = prediction.shape[1]
    kernel = gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    padding = window_size // 2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (
        F.conv2d(prediction * prediction, kernel, padding=padding, groups=channels)
        - mu_x_sq
    )
    sigma_y_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels)
        - mu_y_sq
    )
    sigma_xy = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels)
        - mu_xy
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (
        (mu_x_sq + mu_y_sq + c1)
        * (sigma_x_sq + sigma_y_sq + c2)
    )

    ssim_map = numerator / torch.clamp(
        denominator,
        min=torch.finfo(prediction.dtype).eps,
    )
    return ssim_map.flatten(1).mean(dim=1)


def calculate_batch_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    prediction_f = prediction.float()
    target_f = target.float()

    difference = prediction_f - target_f
    l1 = torch.mean(torch.abs(difference))
    mse = torch.mean(difference.square())

    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
    ssim = calculate_ssim_per_sample(prediction_f, target_f).mean()

    return {
        "l1": float(l1.item()),
        "mse": float(mse.item()),
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
    }


# ============================================================
# 7. History and visualization
# ============================================================


@dataclass(frozen=True)
class HistoryRow:
    epoch: int
    total_loss: float
    l1: float
    mse: float
    psnr: float
    ssim: float
    learning_rate: float
    epoch_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "total_loss": float(self.total_loss),
            "l1": float(self.l1),
            "mse": float(self.mse),
            "psnr": float(self.psnr),
            "ssim": float(self.ssim),
            "learning_rate": float(self.learning_rate),
            "epoch_seconds": float(self.epoch_seconds),
        }


def load_preview_slices(
    dataset: PAMAdjacentYZDataset,
    volume_index: int,
    x_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, str]:
    if not 0 <= volume_index < dataset.num_volumes:
        raise IndexError(
            f"volume_index={volume_index} outside "
            f"0~{dataset.num_volumes - 1}"
        )

    for x_index in x_indices:
        if not 0 <= x_index < X_SIZE:
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

    inputs = np.array(
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

    return inputs, targets, str(pair["volume_key"])


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


def save_prediction_preview(
    model: VanillaUNet,
    dataset: PAMAdjacentYZDataset,
    device: torch.device,
    epoch: int,
) -> None:
    model_was_training = model.training
    model.eval()

    inputs_np, targets_np, volume_key = load_preview_slices(
        dataset=dataset,
        volume_index=SAVE.preview_volume_index,
        x_indices=SAVE.preview_x_indices,
    )

    inputs = torch.from_numpy(inputs_np).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    if TRAIN.use_channels_last and device.type == "cuda":
        inputs = inputs.contiguous(memory_format=torch.channels_last)

    amp_enabled = TRAIN.use_amp and device.type == "cuda"

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            predictions = model(inputs)

    predictions_np = predictions.float().cpu().numpy()

    num_rows = len(SAVE.preview_x_indices)
    fig, axes = plt.subplots(num_rows, 4, figsize=(22, 5.5 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, x_index in enumerate(SAVE.preview_x_indices):
        low_image = np.rot90(inputs_np[row_index, 1], k=-1)
        prediction = np.rot90(predictions_np[row_index, 0], k=-1)
        target = np.rot90(targets_np[row_index, 0], k=-1)
        error = np.abs(target - prediction)

        v_min, v_max = robust_limits([low_image, prediction, target])
        error_max = float(np.percentile(error, 99.8))
        if error_max <= 0:
            error_max = 1e-8

        mse = float(np.mean((prediction - target) ** 2))
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)

        panels = (
            (low_image, "LOW center", "hot", v_min, v_max),
            (prediction, "Vanilla U-Net", "hot", v_min, v_max),
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

        axes[row_index, 1].set_title(
            f"Vanilla U-Net | X={x_index} | PSNR={psnr:.2f} dB"
        )

    fig.suptitle(
        f"Vanilla U-Net | {volume_key} | Epoch {epoch:03d} | "
        "90° clockwise"
    )
    fig.tight_layout()

    save_path = SAVE.preview_dir / f"epoch_{epoch:03d}_samples.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    del inputs
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
    channels = [
        MODEL.base_channels,
        min(MODEL.base_channels * 2, MODEL.max_channels),
        min(MODEL.base_channels * 4, MODEL.max_channels),
        min(MODEL.base_channels * 8, MODEL.max_channels),
        min(MODEL.base_channels * 16, MODEL.max_channels),
    ]

    return {
        "architecture": "VanillaUNet",
        "model_family": "plain 2D U-Net",
        "model_components": (
            "double convolution, max pooling, transposed convolution, "
            "skip concatenation, linear output"
        ),
        "excluded_components": (
            "attention, residual, dense, BEFD, SFA, AFF, edge, wavelet, "
            "GAN, diffusion, multi-stage refinement"
        ),
        "input_shape": list(INPUT_SAMPLE_SHAPE),
        "target_shape": list(TARGET_SAMPLE_SHAPE),
        "data_low_shape": list(DATA.adjacent_low_shape),
        "data_high_shape": list(DATA.high_volume_shape),
        "data_range": "normalized float32, nominally [0,1]",
        "dtype": str(np.dtype(DATA.dtype)),
        "header_size": DATA.header_size,
        "center_channel": 1,
        "input_channels": MODEL.input_channels,
        "output_channels": MODEL.output_channels,
        "base_channels": MODEL.base_channels,
        "max_channels": MODEL.max_channels,
        "channels": channels,
        "use_batch_norm": MODEL.use_batch_norm,
        "output_activation": "linear",
        "training_loss": "L1",
        "logged_metrics": ["L1", "MSE", "PSNR", "SSIM"],
        "optimizer": "Adam",
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
    }


def save_checkpoint(
    model: VanillaUNet,
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

    path.parent.mkdir(parents=True, exist_ok=True)
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

    epochs = [row.epoch for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [row.total_loss for row in history], label="Total/L1 loss")
    plt.plot(epochs, [row.mse for row in history], label="MSE")
    plt.plot(epochs, [1.0 - row.ssim for row in history], label="1 - SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Vanilla U-Net Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE.loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_model_smoke_test(device: torch.device) -> None:
    model = VanillaUNet().to(device)
    model.eval()

    test_input = torch.randn(1, 3, 64, 96, device=device)

    with torch.inference_mode():
        output = model(test_input)

    expected_shape = (1, 1, 64, 96)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Smoke-test output shape {tuple(output.shape)} "
            f"!= {expected_shape}"
        )

    del model
    del test_input
    del output

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# 9. Training
# ============================================================


def train() -> None:
    set_seed(SEED)
    configure_cuda()
    create_output_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = TRAIN.use_amp and device.type == "cuda"

    print("=" * 92)
    print("PURE VANILLA U-NET — PAM 3-ADJACENT Y-Z ENHANCEMENT")
    print("=" * 92)
    print(f"Device                    : {device}")
    if device.type == "cuda":
        print(f"GPU                       : {torch.cuda.get_device_name(0)}")
    print(f"Train LOW                 : {DATA.train_low_dir}")
    print(f"Train HIGH                : {DATA.train_high_dir}")
    print(f"LOW shape                 : {DATA.adjacent_low_shape} [X,C,Y,Z]")
    print(f"HIGH shape                : {DATA.high_volume_shape} [X,Y,Z]")
    print(f"Physical batch size       : {TRAIN.batch_size}")
    print(
        f"Effective batch size      : "
        f"{TRAIN.batch_size * TRAIN.gradient_accumulation_steps}"
    )
    print(f"Slices/volume/epoch       : {TRAIN.slices_per_volume_per_epoch}")
    print(f"Base / max channels       : {MODEL.base_channels} / {MODEL.max_channels}")
    print("Architecture              : plain four-level 2D U-Net")
    print("Output activation         : linear")
    print("Training loss             : L1 only")
    print("Logged metrics            : L1, MSE, PSNR, SSIM")
    print("=" * 92)

    print("\n[1/5] Running Vanilla U-Net smoke test...")
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

    print("\n[3/5] Building pure Vanilla U-Net...")
    model = VanillaUNet().to(device)

    if TRAIN.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_params, trainable_params = count_parameters(model)
    print(f"      Channels            : {model.channels}")
    print(f"      Total parameters    : {total_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=TRAIN.weight_decay,
    )

    scaler = create_grad_scaler(enabled=amp_enabled)

    print("\n[4/5] Starting training...")

    history: list[HistoryRow] = []

    try:
        for epoch in range(1, TRAIN.num_epochs + 1):
            epoch_start = time.time()
            sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            running_l1 = 0.0
            running_mse = 0.0
            running_psnr = 0.0
            running_ssim = 0.0
            seen_batches = 0
            accumulation_counter = 0

            progress = tqdm(
                loader,
                desc=f"Epoch {epoch:03d}/{TRAIN.num_epochs:03d}",
                dynamic_ncols=True,
            )

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
                    targets = targets.contiguous(memory_format=torch.channels_last)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    predictions = model(inputs)
                    total_loss = F.l1_loss(predictions, targets)
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

                with torch.no_grad():
                    metrics = calculate_batch_metrics(predictions, targets)

                running_loss += float(total_loss.detach().item())
                running_l1 += metrics["l1"]
                running_mse += metrics["mse"]
                running_psnr += metrics["psnr"]
                running_ssim += metrics["ssim"]
                seen_batches += 1

                if step % TRAIN.log_every == 0 or step == len(loader):
                    progress.set_postfix(
                        L1=f"{running_l1 / seen_batches:.6f}",
                        PSNR=f"{running_psnr / seen_batches:.2f}",
                        SSIM=f"{running_ssim / seen_batches:.4f}",
                    )

            epoch_seconds = time.time() - epoch_start
            denominator = max(seen_batches, 1)

            row = HistoryRow(
                epoch=epoch,
                total_loss=running_loss / denominator,
                l1=running_l1 / denominator,
                mse=running_mse / denominator,
                psnr=running_psnr / denominator,
                ssim=running_ssim / denominator,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                epoch_seconds=epoch_seconds,
            )
            history.append(row)

            print(
                f"Epoch {epoch:03d} finished | "
                f"L1={row.l1:.8f} | "
                f"MSE={row.mse:.8f} | "
                f"PSNR={row.psnr:.4f} dB | "
                f"SSIM={row.ssim:.6f} | "
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
                print(f"      Saving preview for epoch {epoch:03d}...")
                save_prediction_preview(
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
# 10. Entry point
# ============================================================


if __name__ == "__main__":
    train()
