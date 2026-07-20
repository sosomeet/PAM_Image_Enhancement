# -*- coding: utf-8 -*-
"""
Minimal conditional GAN for PAM image enhancement.

Purpose
-------
Convert paired 3-adjacent LOW Y-Z slices into one HIGH Y-Z slice.

Input
-----
LOW  : [B, 3, 200, 512]
       channel 0 = x-1, channel 1 = center x, channel 2 = x+1

Target
------
HIGH : [B, 1, 200, 512]

GAN structure
-------------
Generator     : plain U-Net encoder-decoder
Discriminator : conditional PatchGAN

Loss
----
Discriminator:
    0.5 * (BCE(real, 1) + BCE(fake, 0))

Generator:
    BCE(D(LOW, G(LOW)), 1) + lambda_L1 * L1(G(LOW), HIGH)

Removed from the previous script
--------------------------------
- residual edge branch
- Haar wavelet branch
- gated detail fusion
- two-stage refinement
- learnable residual scales
- SSIM loss
- boundary loss
- intensity-weighted loss
- intensity-statistics loss
- deep supervision

This is a training-only script. No validation or test split is used.
"""

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

        # Stored binary volume shapes.
        self.high_volume_shape = (200, 200, 512)      # [X,Y,Z]
        self.adjacent_low_shape = (200, 3, 200, 512)  # [X,C,Y,Z]

        self.input_channels = 3
        self.output_channels = 1


class ModelConfig:
    def __init__(self) -> None:
        self.in_channels = 3
        self.out_channels = 1

        # RTX 3060 Ti-oriented widths.
        self.generator_base_channels = 32
        self.generator_max_channels = 256
        self.discriminator_base_channels = 32

        # HIGH data are assumed to be normalized to [0, 1].
        # Sigmoid therefore gives a directly interpretable prediction range.
        self.generator_output_activation = "sigmoid"


class TrainConfig:
    def __init__(self) -> None:
        self.num_epochs = 50

        # Standard pix2pix/cGAN optimizer settings.
        self.generator_learning_rate = 2e-4
        self.discriminator_learning_rate = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

        # Generator objective = adversarial + lambda_L1 * reconstruction.
        # lambda_L1=100 is the conventional paired image-to-image baseline.
        self.lambda_l1 = 100.0

        # 130 volumes x 32 X positions = 4,160 samples/epoch.
        self.slices_per_volume_per_epoch = 32

        # A GAN keeps both G and D in memory. Batch 2 is a safer 8 GB default.
        self.batch_size = 2

        self.num_workers = 4
        self.prefetch_factor = 2
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = True

        self.use_amp = True
        self.use_channels_last = True

        self.log_every = 20
        self.grad_clip_norm = 1.0


class SaveConfig:
    def __init__(self) -> None:
        self.run_root = Path("./14_Pure_Conditional_GAN")

        self.checkpoint_every = 10
        self.map_save_every = 5
        self.map_volume_index = 0
        self.map_batch_size = 4
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
# 4. Dataset / sampler
# ============================================================


class PAMAdjacentYZDataset(Dataset):
    """
    LOW file  : [X,C,Y,Z] = [200,3,200,512]
    HIGH file : [X,Y,Z]   = [200,200,512]

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
        if index < 0 or index >= len(self):
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
    """Sample a fixed number of unique X positions from each volume per epoch."""

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
                f"slices_per_volume={slices_per_volume} exceeds "
                f"X positions={dataset.samples_per_volume}."
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
# 5. GAN model
# ============================================================


def _instance_norm(channels: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(
        channels,
        affine=True,
        track_running_stats=False,
    )


class EncoderBlock(nn.Module):
    """4x4 stride-2 convolution used by the plain U-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not use_norm,
            )
        ]

        if use_norm:
            layers.append(_instance_norm(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """4x4 stride-2 transposed convolution used by the plain U-Net decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            _instance_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """
    Plain four-level U-Net generator.

    No attention, dense connection, residual branch, edge branch, wavelet
    branch, multistage refinement, or auxiliary head is used.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        max_channels: int = 256,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()

        if in_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {in_channels}.")
        if out_channels != 1:
            raise ValueError(f"Expected 1 output channel, got {out_channels}.")
        if output_activation not in {"sigmoid", "tanh", "linear"}:
            raise ValueError(
                "output_activation must be 'sigmoid', 'tanh', or 'linear'."
            )

        c1 = base_channels
        c2 = min(base_channels * 2, max_channels)
        c3 = min(base_channels * 4, max_channels)
        c4 = min(base_channels * 8, max_channels)

        self.output_activation = output_activation

        self.enc1 = EncoderBlock(in_channels, c1, use_norm=False)
        self.enc2 = EncoderBlock(c1, c2, use_norm=True)
        self.enc3 = EncoderBlock(c2, c3, use_norm=True)
        self.enc4 = EncoderBlock(c3, c4, use_norm=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            _instance_norm(c4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            _instance_norm(c4),
            nn.ReLU(inplace=True),
        )

        self.dec3 = DecoderBlock(c4, c3)
        self.dec2 = DecoderBlock(c3 + c3, c2)
        self.dec1 = DecoderBlock(c2 + c2, c1)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                c1 + c1,
                c1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            _instance_norm(c1),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    @staticmethod
    def _pad_to_multiple(
        x: torch.Tensor,
        multiple: int = 16,
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

    def _activate_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                f"Generator expects [B,3,H,W], got {tuple(x.shape)}."
            )

        original_shape = x.shape[-2:]
        x, padding = self._pad_to_multiple(x, multiple=16)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d3 = self.dec3(b)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        output = self.output_conv(self.final_up(d1))
        output = self._activate_output(output)
        output = self._remove_padding(output, padding)

        if output.shape[-2:] != original_shape:
            raise RuntimeError(
                f"Generator output shape {tuple(output.shape[-2:])} "
                f"does not match input shape {tuple(original_shape)}."
            )

        return output


class PatchDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator.

    It judges a pair rather than a single image:
        concat(LOW[3 channels], candidate HIGH[1 channel]) -> patch logits
    """

    def __init__(
        self,
        condition_channels: int = 3,
        image_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        in_channels = condition_channels + image_channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                c1,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c1,
                c2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            _instance_norm(c2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c2,
                c3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            _instance_norm(c3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c3,
                c4,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            _instance_norm(c4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                c4,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    def forward(
        self,
        condition: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if condition.ndim != 4 or condition.size(1) != 3:
            raise ValueError(
                f"Condition must be [B,3,H,W], got {tuple(condition.shape)}."
            )
        if image.ndim != 4 or image.size(1) != 1:
            raise ValueError(
                f"Image must be [B,1,H,W], got {tuple(image.shape)}."
            )
        if condition.shape[0] != image.shape[0]:
            raise ValueError("Condition and image batch sizes differ.")
        if condition.shape[-2:] != image.shape[-2:]:
            raise ValueError("Condition and image spatial sizes differ.")

        pair = torch.cat([condition, image], dim=1)
        return self.net(pair)


# Compatibility aliases for evaluators that import Model/UNet/PAMEnhancer.
Generator = UNetGenerator
PAMEnhancer = UNetGenerator
UNet = UNetGenerator
Model = UNetGenerator
Discriminator = PatchDiscriminator


def initialize_gan_weights(module: nn.Module) -> None:
    """Standard DCGAN/pix2pix initialization."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm2d) and module.affine:
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


# ============================================================
# 6. GAN losses
# ============================================================


class ConditionalGANLoss(nn.Module):
    """Minimal BCE adversarial loss plus L1 reconstruction loss."""

    def __init__(self, lambda_l1: float = 100.0) -> None:
        super().__init__()

        if lambda_l1 < 0:
            raise ValueError("lambda_l1 must be >= 0.")

        self.lambda_l1 = float(lambda_l1)
        self.adversarial = nn.BCEWithLogitsLoss()
        self.reconstruction = nn.L1Loss()

    def discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        real_loss = self.adversarial(
            real_logits,
            torch.ones_like(real_logits),
        )
        fake_loss = self.adversarial(
            fake_logits,
            torch.zeros_like(fake_logits),
        )
        total = 0.5 * (real_loss + fake_loss)

        return total, {
            "d_real": real_loss.detach(),
            "d_fake": fake_loss.detach(),
        }

    def generator_loss(
        self,
        fake_logits: torch.Tensor,
        fake_image: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        adversarial_loss = self.adversarial(
            fake_logits,
            torch.ones_like(fake_logits),
        )
        l1_loss = self.reconstruction(fake_image, target)
        total = adversarial_loss + self.lambda_l1 * l1_loss

        return total, {
            "g_adversarial": adversarial_loss.detach(),
            "g_l1": l1_loss.detach(),
        }


# ============================================================
# 7. History
# ============================================================


class HistoryRow:
    fieldnames = [
        "epoch",
        "generator_total",
        "generator_adversarial",
        "generator_l1",
        "discriminator_total",
        "discriminator_real",
        "discriminator_fake",
        "generator_learning_rate",
        "discriminator_learning_rate",
        "epoch_seconds",
    ]

    def __init__(
        self,
        epoch: int,
        generator_total: float,
        generator_adversarial: float,
        generator_l1: float,
        discriminator_total: float,
        discriminator_real: float,
        discriminator_fake: float,
        generator_learning_rate: float,
        discriminator_learning_rate: float,
        epoch_seconds: float,
    ) -> None:
        self.epoch = int(epoch)
        self.generator_total = float(generator_total)
        self.generator_adversarial = float(generator_adversarial)
        self.generator_l1 = float(generator_l1)
        self.discriminator_total = float(discriminator_total)
        self.discriminator_real = float(discriminator_real)
        self.discriminator_fake = float(discriminator_fake)
        self.generator_learning_rate = float(generator_learning_rate)
        self.discriminator_learning_rate = float(discriminator_learning_rate)
        self.epoch_seconds = float(epoch_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.fieldnames
        }


# ============================================================
# 8. MAP visualization
# ============================================================


def load_full_training_volume(
    dataset: PAMAdjacentYZDataset,
    volume_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if volume_index < 0 or volume_index >= dataset.num_volumes:
        raise IndexError(
            f"volume_index={volume_index} outside "
            f"0~{dataset.num_volumes - 1}."
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
    v_min = float(np.min(values))
    v_max = float(np.percentile(values, upper_percentile))

    if v_max <= v_min:
        v_max = v_min + 1e-8

    return v_min, v_max


def save_training_map(
    generator: nn.Module,
    dataset: PAMAdjacentYZDataset,
    device: torch.device,
    epoch: int,
) -> None:
    was_training = generator.training
    generator.eval()

    low_volume, high_volume = load_full_training_volume(
        dataset,
        SAVE.map_volume_index,
    )
    low_center = low_volume[:, 1]

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
                prediction = generator(batch)

            prediction_volume[start:end] = (
                prediction[:, 0].float().cpu().numpy()
            )

    low_map = np.max(low_center, axis=0)
    pred_map = np.max(prediction_volume, axis=0)
    high_map = np.max(high_volume, axis=0)
    error_map = np.abs(high_map - pred_map)

    # Existing project convention: rotate 90 degrees clockwise.
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
        (pred_map, "GAN Prediction Y-Z MAP", "hot", v_min, v_max),
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
        f"Minimal Conditional GAN PAM Enhancement | Epoch {epoch:03d} | "
        "90° clockwise"
    )
    fig.tight_layout()

    save_path = SAVE.train_map_dir / f"epoch_{epoch:03d}_map.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    del low_volume
    del high_volume
    del prediction_volume

    if was_training:
        generator.train()


# ============================================================
# 9. Checkpoint / history utilities
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
        "architecture": "MinimalConditionalGAN",
        "generator_architecture": "PlainFourLevelUNetGenerator",
        "discriminator_architecture": "ConditionalPatchGAN",
        "input_shape": list(INPUT_SAMPLE_SHAPE),
        "target_shape": list(TARGET_SAMPLE_SHAPE),
        "data_low_shape": list(DATA.adjacent_low_shape),
        "data_high_shape": list(DATA.high_volume_shape),
        "dtype": str(np.dtype(DATA.dtype)),
        "header_size": DATA.header_size,
        "center_channel": 1,
        "generator_base_channels": MODEL.generator_base_channels,
        "generator_max_channels": MODEL.generator_max_channels,
        "discriminator_base_channels": MODEL.discriminator_base_channels,
        "generator_output_activation": MODEL.generator_output_activation,
        "loss": "BCE adversarial + lambda_L1 * L1 reconstruction",
        "lambda_l1": TRAIN.lambda_l1,
        "generator_optimizer": "Adam",
        "discriminator_optimizer": "Adam",
        "generator_learning_rate": TRAIN.generator_learning_rate,
        "discriminator_learning_rate": TRAIN.discriminator_learning_rate,
        "adam_beta1": TRAIN.beta1,
        "adam_beta2": TRAIN.beta2,
        "batch_size": TRAIN.batch_size,
        "slices_per_volume_per_epoch": TRAIN.slices_per_volume_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "use_amp": TRAIN.use_amp,
        "use_channels_last": TRAIN.use_channels_last,
    }


def save_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    scaler_g: torch.cuda.amp.GradScaler,
    scaler_d: torch.cuda.amp.GradScaler,
    epoch: int,
    history_row: HistoryRow,
    samples_per_epoch: int,
    path: Path,
) -> None:
    generator_state = generator.state_dict()

    checkpoint = {
        "epoch": epoch,

        # model_state_dict is retained for compatibility with evaluators that
        # expect a single enhancement network state dictionary.
        "model_state_dict": generator_state,
        "generator_state_dict": generator_state,
        "discriminator_state_dict": discriminator.state_dict(),

        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "scaler_g_state_dict": scaler_g.state_dict(),
        "scaler_d_state_dict": scaler_d.state_dict(),
        "history_row": history_row.to_dict(),
        **model_metadata(samples_per_epoch),
    }
    torch.save(checkpoint, path)


def save_training_history(history: list[HistoryRow]) -> None:
    with SAVE.history_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HistoryRow.fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in history)


def save_loss_curve(history: list[HistoryRow]) -> None:
    if not history:
        return

    epochs = [row.epoch for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        [row.generator_adversarial for row in history],
        marker="o",
        markersize=3,
        label="Generator adversarial",
    )
    plt.plot(
        epochs,
        [row.generator_l1 for row in history],
        marker="o",
        markersize=3,
        label="Generator L1",
    )
    plt.plot(
        epochs,
        [row.discriminator_total for row in history],
        marker="o",
        markersize=3,
        label="Discriminator total",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Minimal Conditional GAN Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE.loss_curve_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 10. Smoke test
# ============================================================


def run_model_smoke_test(device: torch.device) -> None:
    generator = UNetGenerator(
        in_channels=MODEL.in_channels,
        out_channels=MODEL.out_channels,
        base_channels=MODEL.generator_base_channels,
        max_channels=MODEL.generator_max_channels,
        output_activation=MODEL.generator_output_activation,
    ).to(device)

    discriminator = PatchDiscriminator(
        condition_channels=MODEL.in_channels,
        image_channels=MODEL.out_channels,
        base_channels=MODEL.discriminator_base_channels,
    ).to(device)

    if TRAIN.use_channels_last and device.type == "cuda":
        generator = generator.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)

    dummy_input = torch.randn(1, 3, 64, 128, device=device)
    dummy_target = torch.rand(1, 1, 64, 128, device=device)

    if TRAIN.use_channels_last and device.type == "cuda":
        dummy_input = dummy_input.contiguous(memory_format=torch.channels_last)
        dummy_target = dummy_target.contiguous(memory_format=torch.channels_last)

    generator.eval()
    discriminator.eval()

    with torch.inference_mode():
        fake = generator(dummy_input)
        real_logits = discriminator(dummy_input, dummy_target)
        fake_logits = discriminator(dummy_input, fake)

    if fake.shape != dummy_target.shape:
        raise RuntimeError(
            f"Smoke test generator shape failed: {tuple(fake.shape)}."
        )
    if real_logits.shape != fake_logits.shape:
        raise RuntimeError("Smoke test discriminator patch shapes differ.")
    if not torch.isfinite(fake).all():
        raise RuntimeError("Smoke test generator output contains NaN/Inf.")
    if not torch.isfinite(real_logits).all():
        raise RuntimeError("Smoke test discriminator output contains NaN/Inf.")

    del generator
    del discriminator
    del dummy_input
    del dummy_target
    del fake
    del real_logits
    del fake_logits

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
    print("PAM ENHANCEMENT — MINIMAL CONDITIONAL GAN")
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
    print(f"Batch size                : {TRAIN.batch_size}")
    print(f"Slices/volume/epoch       : {TRAIN.slices_per_volume_per_epoch}")
    print(
        f"Generator channels        : "
        f"{MODEL.generator_base_channels} -> {MODEL.generator_max_channels}"
    )
    print(f"Discriminator base        : {MODEL.discriminator_base_channels}")
    print(f"Generator output          : {MODEL.generator_output_activation}")
    print(
        "Generator loss            : "
        f"BCE adversarial + {TRAIN.lambda_l1:.1f} * L1"
    )
    print("Discriminator loss        : 0.5 * (BCE real + BCE fake)")
    print("=" * 92)

    print("\n[1/5] Running GAN smoke test...")
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

    print("\n[3/5] Building generator and discriminator...")
    generator = UNetGenerator(
        in_channels=MODEL.in_channels,
        out_channels=MODEL.out_channels,
        base_channels=MODEL.generator_base_channels,
        max_channels=MODEL.generator_max_channels,
        output_activation=MODEL.generator_output_activation,
    ).to(device)

    discriminator = PatchDiscriminator(
        condition_channels=MODEL.in_channels,
        image_channels=MODEL.out_channels,
        base_channels=MODEL.discriminator_base_channels,
    ).to(device)

    generator.apply(initialize_gan_weights)
    discriminator.apply(initialize_gan_weights)

    if TRAIN.use_channels_last and device.type == "cuda":
        generator = generator.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)

    g_total_params, g_trainable_params = count_parameters(generator)
    d_total_params, d_trainable_params = count_parameters(discriminator)

    print(f"      Generator parameters    : {g_total_params:,}")
    print(f"      Generator trainable     : {g_trainable_params:,}")
    print(f"      Discriminator parameters: {d_total_params:,}")
    print(f"      Discriminator trainable : {d_trainable_params:,}")

    criterion = ConditionalGANLoss(lambda_l1=TRAIN.lambda_l1)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=TRAIN.generator_learning_rate,
        betas=(TRAIN.beta1, TRAIN.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=TRAIN.discriminator_learning_rate,
        betas=(TRAIN.beta1, TRAIN.beta2),
    )

    amp_enabled = TRAIN.use_amp and device.type == "cuda"
    scaler_g = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    scaler_d = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print("\n[4/5] Starting GAN training...")

    history: list[HistoryRow] = []

    if SAVE.save_initial_map:
        save_training_map(
            generator=generator,
            dataset=dataset,
            device=device,
            epoch=0,
        )

    try:
        for epoch in range(1, TRAIN.num_epochs + 1):
            epoch_start = time.time()
            sampler.set_epoch(epoch)

            generator.train()
            discriminator.train()

            running_g_total = 0.0
            running_g_adv = 0.0
            running_g_l1 = 0.0
            running_d_total = 0.0
            running_d_real = 0.0
            running_d_fake = 0.0
            seen_batches = 0

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

                # ----------------------------------------------------
                # One generator forward pass is reused for D and G.
                # D receives fake.detach(), so D backward does not alter G.
                # ----------------------------------------------------
                optimizer_g.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    fake_images = generator(inputs)

                # ----------------------------------------------------
                # 1) Discriminator update
                # ----------------------------------------------------
                set_requires_grad(discriminator, True)
                optimizer_d.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    real_logits = discriminator(inputs, targets)
                    fake_logits_d = discriminator(inputs, fake_images.detach())
                    d_loss, d_parts = criterion.discriminator_loss(
                        real_logits,
                        fake_logits_d,
                    )

                scaler_d.scale(d_loss).backward()
                scaler_d.unscale_(optimizer_d)

                if TRAIN.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(),
                        max_norm=TRAIN.grad_clip_norm,
                    )

                scaler_d.step(optimizer_d)
                scaler_d.update()

                # ----------------------------------------------------
                # 2) Generator update
                # ----------------------------------------------------
                set_requires_grad(discriminator, False)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    fake_logits_g = discriminator(inputs, fake_images)
                    g_loss, g_parts = criterion.generator_loss(
                        fake_logits=fake_logits_g,
                        fake_image=fake_images,
                        target=targets,
                    )

                scaler_g.scale(g_loss).backward()
                scaler_g.unscale_(optimizer_g)

                if TRAIN.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        generator.parameters(),
                        max_norm=TRAIN.grad_clip_norm,
                    )

                scaler_g.step(optimizer_g)
                scaler_g.update()
                set_requires_grad(discriminator, True)

                running_g_total += float(g_loss.detach().item())
                running_g_adv += float(g_parts["g_adversarial"].item())
                running_g_l1 += float(g_parts["g_l1"].item())
                running_d_total += float(d_loss.detach().item())
                running_d_real += float(d_parts["d_real"].item())
                running_d_fake += float(d_parts["d_fake"].item())
                seen_batches += 1

                if step % TRAIN.log_every == 0 or step == len(loader):
                    denominator = max(seen_batches, 1)
                    progress.set_postfix(
                        G=f"{running_g_total / denominator:.4f}",
                        G_adv=f"{running_g_adv / denominator:.4f}",
                        L1=f"{running_g_l1 / denominator:.5f}",
                        D=f"{running_d_total / denominator:.4f}",
                    )

            denominator = max(seen_batches, 1)
            epoch_seconds = time.time() - epoch_start

            row = HistoryRow(
                epoch=epoch,
                generator_total=running_g_total / denominator,
                generator_adversarial=running_g_adv / denominator,
                generator_l1=running_g_l1 / denominator,
                discriminator_total=running_d_total / denominator,
                discriminator_real=running_d_real / denominator,
                discriminator_fake=running_d_fake / denominator,
                generator_learning_rate=float(optimizer_g.param_groups[0]["lr"]),
                discriminator_learning_rate=float(optimizer_d.param_groups[0]["lr"]),
                epoch_seconds=epoch_seconds,
            )
            history.append(row)

            print(
                f"Epoch {epoch:03d} finished | "
                f"G={row.generator_total:.6f} | "
                f"G_adv={row.generator_adversarial:.6f} | "
                f"G_L1={row.generator_l1:.8f} | "
                f"D={row.discriminator_total:.6f} | "
                f"D_real={row.discriminator_real:.6f} | "
                f"D_fake={row.discriminator_fake:.6f} | "
                f"time={epoch_seconds / 60.0:.2f} min"
            )

            save_checkpoint(
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scaler_g=scaler_g,
                scaler_d=scaler_d,
                epoch=epoch,
                history_row=row,
                samples_per_epoch=samples_per_epoch,
                path=SAVE.latest_model_path,
            )

            if epoch % SAVE.checkpoint_every == 0:
                save_checkpoint(
                    generator=generator,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    scaler_g=scaler_g,
                    scaler_d=scaler_d,
                    epoch=epoch,
                    history_row=row,
                    samples_per_epoch=samples_per_epoch,
                    path=SAVE.checkpoint_dir / f"epoch_{epoch:03d}.pth",
                )

            save_training_history(history)
            save_loss_curve(history)

            if epoch % SAVE.map_save_every == 0:
                print(f"      Saving MAP for epoch {epoch:03d}...")
                save_training_map(
                    generator=generator,
                    dataset=dataset,
                    device=device,
                    epoch=epoch,
                )

        print("\n[5/5] Saving final GAN checkpoint...")

        final_row = history[-1]
        save_checkpoint(
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
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

    finally:
        dataset.close()


# ============================================================
# 12. Entry point
# ============================================================


if __name__ == "__main__":
    train()
