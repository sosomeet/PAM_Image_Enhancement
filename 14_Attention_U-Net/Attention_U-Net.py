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
    """Pre-generated 3-adjacent Y-Z PAM data configuration."""

    root: Path = Path("./data")
    header_size: int = 48
    dtype: type[np.float32] = np.float32

    # HIGH: [X, Y, Z]
    high_volume_shape: tuple[int, int, int] = (200, 200, 512)

    # LOW: [X, C, Y, Z], where C contains X-1, X, X+1 planes
    adjacent_low_shape: tuple[int, int, int, int] = (200, 3, 200, 512)

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

    # Kept identical to the source experiment so that the architecture
    # comparison is not confounded by a different training objective.
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
    """
    Plain 2D Attention U-Net configuration.

    Included:
      - Standard double-convolution U-Net blocks
      - Additive grid attention gates on skip connections

    Excluded:
      - Dual path
      - Fully dense blocks
      - SFA
      - BEFD
      - AFF
      - Residual blocks
      - Deep supervision
    """

    input_channels: int = 3
    output_channels: int = 1
    base_channels: int = 8
    max_channels: int = 128
    use_batch_norm: bool = True

    # The paper states that the first, shallowest skip connection is not gated.
    attention_stages: tuple[str, ...] = ("encoder4", "encoder3", "encoder2")

    # Initial sigmoid coefficient ~= 0.88, so gates begin close to pass-through.
    attention_pass_bias: float = 2.0

    # Y=200 is padded to 208 so that four 2x pooling operations are exact.
    pad_top: int = 4
    pad_bottom: int = 4


@dataclass(frozen=True)
class SaveConfig:
    experiment_root: Path = Path("./14_Attention_U-Net")
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
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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

    def _open_memmap(self, volume_index: int, *, is_low: bool) -> np.memmap:
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

        if input_array.shape != DATA.input_sample_shape:
            raise ValueError(
                f"Input shape mismatch for {pair.volume_key}, X={x_index}: "
                f"{input_array.shape} != {DATA.input_sample_shape}"
            )
        if target_array.shape != DATA.target_sample_shape:
            raise ValueError(
                f"Target shape mismatch for {pair.volume_key}, X={x_index}: "
                f"{target_array.shape} != {DATA.target_sample_shape}"
            )

        return {
            "input": torch.from_numpy(input_array),
            "target": torch.from_numpy(target_array),
            "volume_key": pair.volume_key,
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
    """Deterministic random X-slice subset for each volume and epoch."""

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
            x_indices.sort()
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

    loader_kwargs: dict[str, Any] = {
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
        loader_kwargs["prefetch_factor"] = TRAIN.prefetch_factor

    return dataset, sampler, DataLoader(**loader_kwargs)


# ============================================================
# Plain Attention U-Net model
# ============================================================


class DoubleConv(nn.Module):
    """Standard U-Net block: (3x3 Conv -> BN -> ReLU) x2."""

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


class AdditiveAttentionGate(nn.Module):
    """
    Paper-style additive grid attention gate.

    x: encoder skip feature at a finer spatial scale
    g: decoder/context feature at the next coarser spatial scale

    The skip feature is projected with a stride-2 1x1 convolution to the
    gating resolution. The resulting scalar attention map is then resampled
    back to the skip resolution and multiplied element-wise with x.
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        intermediate_channels: int,
    ) -> None:
        super().__init__()

        if min(skip_channels, gate_channels, intermediate_channels) <= 0:
            raise ValueError("Attention gate channel counts must be positive.")

        self.theta_x = nn.Conv2d(
            skip_channels,
            intermediate_channels,
            kernel_size=1,
            stride=2,
            padding=0,
            bias=True,
        )
        self.phi_g = nn.Conv2d(
            gate_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.psi = nn.Conv2d(
            intermediate_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Start close to an identity/pass-through gate, as described in the paper.
        nn.init.zeros_(self.psi.weight)
        nn.init.constant_(self.psi.bias, MODEL.attention_pass_bias)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)

        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        attention_logits = self.psi(self.relu(theta_x + phi_g))
        attention = self.sigmoid(attention_logits)
        attention = F.interpolate(
            attention,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return x * attention


class AttentionUNet2D(nn.Module):
    """
    Plain 2D Attention U-Net adapted for PAM image enhancement.

    Encoder/decoder:
        3 -> 8 -> 16 -> 32 -> 64 -> 128 -> 64 -> 32 -> 16 -> 8 -> 1

    Attention gates:
        encoder4 skip, gated by bottleneck
        encoder3 skip, gated by decoder4
        encoder2 skip, gated by decoder3

    The highest-resolution encoder1 skip remains ungated, following the
    original paper's statement that the first low-level skip is not gated.
    """

    def __init__(
        self,
        in_channels: int = MODEL.input_channels,
        out_channels: int = MODEL.output_channels,
        base_channels: int = MODEL.base_channels,
    ) -> None:
        super().__init__()

        c1, c2, c3, c4, c5 = [base_channels * (2**index) for index in range(5)]
        if c5 != MODEL.max_channels:
            raise ValueError(
                f"Channel configuration mismatch: bottleneck={c5}, "
                f"configured max={MODEL.max_channels}"
            )

        self.encoder1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(c4, c5)

        self.attention4 = AdditiveAttentionGate(
            skip_channels=c4,
            gate_channels=c5,
            intermediate_channels=max(1, c4 // 2),
        )
        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(c4 * 2, c4)

        self.attention3 = AdditiveAttentionGate(
            skip_channels=c3,
            gate_channels=c4,
            intermediate_channels=max(1, c3 // 2),
        )
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(c3 * 2, c3)

        self.attention2 = AdditiveAttentionGate(
            skip_channels=c2,
            gate_channels=c3,
            intermediate_channels=max(1, c2 // 2),
        )
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(c2 * 2, c2)

        # First low-level skip is intentionally not gated.
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(c1 * 2, c1)

        # Linear output for regression/image enhancement.
        self.output_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    @staticmethod
    def _concat(decoder_feature: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        if decoder_feature.shape[-2:] != skip_feature.shape[-2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=skip_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.cat([skip_feature, decoder_feature], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            pad=(0, 0, MODEL.pad_top, MODEL.pad_bottom),
            mode="reflect",
        )

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))

        gated_e4 = self.attention4(e4, bottleneck)
        d4 = self.upconv4(bottleneck)
        d4 = self.decoder4(self._concat(d4, gated_e4))

        gated_e3 = self.attention3(e3, d4)
        d3 = self.upconv3(d4)
        d3 = self.decoder3(self._concat(d3, gated_e3))

        gated_e2 = self.attention2(e2, d3)
        d2 = self.upconv2(d3)
        d2 = self.decoder2(self._concat(d2, gated_e2))

        d1 = self.upconv1(d2)
        d1 = self.decoder1(self._concat(d1, e1))

        output = self.output_conv(d1)
        return output[:, :, MODEL.pad_top : -MODEL.pad_bottom, :]


# ============================================================
# Fixed benchmark loss: L1 + SSIM + Sobel edge
# ============================================================


def _gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(
            f"SSIM window_size must be a positive odd integer, got {window_size}"
        )
    if sigma <= 0:
        raise ValueError(f"SSIM sigma must be > 0, got {sigma}")

    coordinates = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
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

        sigma_x_sq = (
            F.conv2d(prediction.square(), window, padding=padding, groups=channels)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.conv2d(target.square(), window, padding=padding, groups=channels)
            - mu_y_sq
        )
        sigma_xy = (
            F.conv2d(prediction * target, window, padding=padding, groups=channels)
            - mu_xy
        )

        numerator = (2.0 * mu_xy + self.c1) * (2.0 * sigma_xy + self.c2)
        denominator = (mu_x_sq + mu_y_sq + self.c1) * (
            sigma_x_sq + sigma_y_sq + self.c2
        )
        ssim_map = numerator / denominator.clamp_min(1e-12)
        return 1.0 - ssim_map.mean()


class SobelEdgeLoss(nn.Module):
    """L1 difference between Sobel gradient magnitudes."""

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
        return F.l1_loss(
            self._gradient_magnitude(prediction),
            self._gradient_magnitude(target),
        )


@dataclass
class LossOutput:
    total: torch.Tensor
    l1: torch.Tensor
    ssim: torch.Tensor
    edge: torch.Tensor


class BenchmarkCompositeLoss(nn.Module):
    """Fixed comparison loss: 0.7 L1 + 0.2 SSIM + 0.1 Sobel edge."""

    def __init__(self) -> None:
        super().__init__()
        total_weight = TRAIN.l1_weight + TRAIN.ssim_weight + TRAIN.edge_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Composite loss weights must sum to 1.0, got {total_weight:.6f}"
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
# Verification and optimizer
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
    optimizer_kwargs = {
        "params": model.parameters(),
        "lr": TRAIN.learning_rate,
        "betas": (0.9, 0.999),
    }

    if device.type == "cuda":
        try:
            optimizer = torch.optim.Adam(**optimizer_kwargs, fused=True)
            print("  Adam implementation    : fused")
            return optimizer
        except (TypeError, RuntimeError):
            pass

    print("  Adam implementation    : standard")
    return torch.optim.Adam(**optimizer_kwargs)


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
    criterion: BenchmarkCompositeLoss,
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
        inputs = move_tensor_to_device(
            batch["input"],
            device,
            channels_last=channels_last,
        )
        targets = move_tensor_to_device(batch["target"], device)
        batch_size = inputs.size(0)

        with autocast_context(device, amp_enabled):
            predictions = model(inputs)
            loss_output = criterion(predictions, targets)
            loss_for_backward = (
                loss_output.total / TRAIN.gradient_accumulation_steps
            )

        scaler.scale(loss_for_backward).backward()

        accumulation_boundary = (
            (step + 1) % TRAIN.gradient_accumulation_steps == 0
        )
        last_batch = step + 1 == len(train_loader)

        if accumulation_boundary or last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_total += loss_output.total.detach().double() * batch_size
        running_l1 += loss_output.l1.detach().double() * batch_size
        running_ssim += loss_output.ssim.detach().double() * batch_size
        running_edge += loss_output.edge.detach().double() * batch_size
        total_samples += batch_size

        if (step + 1) % TRAIN.log_every == 0 or last_batch:
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
# Full-volume MAP visualization
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
    expected_shape = (DATA.x_size, 1, DATA.y_size, DATA.z_size)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"\n{label} reconstruction shape mismatch.\n"
            f"Actual   : {tuple(tensor.shape)}\n"
            f"Expected : {expected_shape}"
        )


def _safe_percentile_max(
    array: np.ndarray,
    percentile: float,
    fallback: float = 1.0,
) -> float:
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
    prediction_slices: list[torch.Tensor] = []
    target_slices: list[torch.Tensor] = []
    volume_key: str | None = None

    try:
        for batch in volume_loader:
            inputs = move_tensor_to_device(
                batch["input"],
                device,
                channels_last=channels_last,
            )
            targets = batch["target"]

            with autocast_context(device, amp_enabled):
                predictions = model(inputs)

            low_slices.append(inputs[:, 1:2].detach().float().cpu())
            prediction_slices.append(
                predictions.clamp_min(0.0).detach().float().cpu()
            )
            target_slices.append(targets.detach().float().cpu())

            if volume_key is None:
                volume_key = batch["volume_key"][0]

        low_tensor = torch.cat(low_slices, dim=0)
        prediction_tensor = torch.cat(prediction_slices, dim=0)
        target_tensor = torch.cat(target_slices, dim=0)

        for label, tensor in (
            ("LOW", low_tensor),
            ("Prediction", prediction_tensor),
            ("Target", target_tensor),
        ):
            _validate_reconstructed_tensor(label, tensor)

        low_volume = low_tensor[:, 0].numpy()
        prediction_volume = prediction_tensor[:, 0].numpy()
        target_volume = target_tensor[:, 0].numpy()

        low_map = np.max(low_volume, axis=2)
        prediction_map = np.max(prediction_volume, axis=2)
        target_map = np.max(target_volume, axis=2)
        error_map = np.abs(prediction_map - target_map)

        display_vmax = _safe_percentile_max(target_map, 99.5)
        error_vmax = _safe_percentile_max(error_map, 99.5)

        figure, axes = plt.subplots(1, 4, figsize=(20, 5))
        panels = (
            (low_map, "LOW MAP", "hot", 0.0, display_vmax),
            (
                prediction_map,
                f"Prediction MAP\nEpoch {epoch}",
                "hot",
                0.0,
                display_vmax,
            ),
            (target_map, "HIGH MAP", "hot", 0.0, display_vmax),
            (error_map, "Absolute Error", "viridis", 0.0, error_vmax),
        )

        for axis, (image, title, colormap, vmin, vmax) in zip(axes, panels):
            shown = axis.imshow(
                image,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(title)
            axis.axis("off")
            figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)

        figure.suptitle(
            f"Plain Attention U-Net Train MAP | {volume_key} | Epoch {epoch}",
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
# Checkpoint and history
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
        "model_name": "AttentionUNet2D",
        "experiment": "Plain Attention U-Net ablation",
        "training_plane": "YZ",
        "adjacent_axis": "X",
        "low_data_shape_order": "[X, C, Y, Z]",
        "high_data_shape_order": "[X, Y, Z]",
        "input_channels": MODEL.input_channels,
        "output_channels": MODEL.output_channels,
        "base_channels": MODEL.base_channels,
        "max_channels": MODEL.max_channels,
        "channel_progression": "3-8-16-32-64-128-64-32-16-8-1",
        "conv_block": "(3x3 Conv + BatchNorm + ReLU) x2",
        "attention_type": "additive grid attention",
        "attention_stages": list(MODEL.attention_stages),
        "attention_first_skip_gated": False,
        "attention_projection": "1x1 Conv; skip projection uses stride 2",
        "attention_normalization": "sigmoid",
        "attention_pass_bias": MODEL.attention_pass_bias,
        "dual_path_enabled": False,
        "fully_dense_enabled": False,
        "sfa_enabled": False,
        "befd_enabled": False,
        "aff_enabled": False,
        "residual_enabled": False,
        "deep_supervision_enabled": False,
        "physical_batch_size": TRAIN.batch_size,
        "gradient_accumulation_steps": TRAIN.gradient_accumulation_steps,
        "effective_batch_size": TRAIN.effective_batch_size,
        "slices_per_volume_per_epoch": TRAIN.slices_per_volume_per_epoch,
        "samples_per_epoch": samples_per_epoch,
        "use_amp": TRAIN.use_amp,
        "use_channels_last": TRAIN.use_channels_last,
        "final_output_activation": "None (Linear)",
        "loss": "0.7*L1 + 0.2*(1-SSIM) + 0.1*SobelEdge",
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
    fieldnames = (
        list(asdict(history[0]).keys())
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
    plt.title("Plain Attention U-Net Training Loss")
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
    print("PAM 3-ADJACENT Y-Z PLAIN ATTENTION U-NET TRAINING")
    print("TRAIN-ONLY | PRE-GENERATED 3-ADJACENT LOW DATA")
    print("=" * 70)
    print(f"Device                  : {device}")

    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU                     : {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU memory        : "
            f"{properties.total_memory / (1024**3):.2f} GB"
        )


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
        ("Architecture", "Plain 2D Attention U-Net"),
        ("Task", "PAM image enhancement regression"),
        ("Training plane", "Y-Z"),
        ("Adjacent axis", "X"),
        ("Channels", "3-8-16-32-64-128-64-32-16-8-1"),
        ("Parameters", f"{count_trainable_parameters(model):,}"),
        ("Conv block", "Conv-BN-ReLU x2"),
        ("Attention", "Additive grid attention"),
        ("Gated skips", ", ".join(MODEL.attention_stages)),
        ("First skip", "Direct concatenation; no gate"),
        ("Dual path", "Removed"),
        ("Fully dense", "Removed"),
        ("SFA / BEFD / AFF", "Removed"),
        ("Residual blocks", "Removed"),
        ("Deep supervision", "Removed"),
        ("Final activation", "None (Linear)"),
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
        ("Loss purpose", "Fixed for architecture-only comparison"),
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

        model = AttentionUNet2D().to(device)
        if use_channels_last(device):
            model = model.to(memory_format=torch.channels_last)
        print_model_summary(model)

        criterion = BenchmarkCompositeLoss()
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
                model=model,
                train_dataset=train_dataset,
                device=device,
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
                    model=model,
                    train_dataset=train_dataset,
                    device=device,
                    epoch=epoch,
                    amp_enabled=amp_enabled,
                    volume_index=SAVE.map_volume_index,
                )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_metrics.total,
                samples_per_epoch=samples_per_epoch,
                path=SAVE.latest_model_path,
            )

            if epoch % SAVE.checkpoint_every == 0:
                checkpoint_path = SAVE.checkpoint_dir / f"epoch_{epoch:03d}.pth"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_metrics.total,
                    samples_per_epoch=samples_per_epoch,
                    path=checkpoint_path,
                )
                print(f"Saved checkpoint: {checkpoint_path}")

            save_training_history(history)
            save_loss_curve(history)

        final_train_loss = history[-1].train_total_loss
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=TRAIN.num_epochs,
            train_loss=final_train_loss,
            samples_per_epoch=samples_per_epoch,
            path=SAVE.final_model_path,
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
