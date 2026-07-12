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

DATA_ROOT = Path("./data")
TRAIN_LOW_DIR = DATA_ROOT / "3d_Train" / "LOW"
TRAIN_HIGH_DIR = DATA_ROOT / "norm_Train" / "HIGH"

# ------------------------------------------------------------
# Binary data specification
# ------------------------------------------------------------

HEADER_SIZE = 48
DATA_DTYPE = np.float32

# HIGH: [X, Y, Z]
HIGH_VOLUME_SHAPE = (200, 200, 512)

# 3-adjacent LOW: [Y, C, X, Z]
ADJACENT_LOW_SHAPE = (200, 3, 200, 512)

X_SIZE = 200
Y_SIZE = 200
Z_SIZE = 512
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

INPUT_SAMPLE_SHAPE = (3, 200, 512)
TARGET_SAMPLE_SHAPE = (1, 200, 512)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_EPOCHS = 50
LEARNING_RATE = 2e-4

# Each epoch uses only this many Y slices per volume.
# 130 volumes x 32 slices = 4,160 training samples / epoch.
SLICES_PER_VOLUME_PER_EPOCH = 32

# Recommended starting point for RTX 3060 Ti 8GB.
# Reduce to 4 if OOM occurs. Increase to 12/16 only after checking VRAM.
BATCH_SIZE = 8

# No gradient accumulation by default.
GRADIENT_ACCUMULATION_STEPS = 1

# Windows/local workstation starting point.
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0

# Mixed precision
USE_AMP = True

# Channels-last often improves 2D convolution throughput on Ampere GPUs.
USE_CHANNELS_LAST = True

# Update tqdm postfix only every N steps to reduce CPU/GPU synchronization.
LOG_EVERY = 20

# ------------------------------------------------------------
# U-Net
# ------------------------------------------------------------

BASE_CHANNELS = 8
MAX_CHANNELS = 128

# Input height 200 is not divisible by 16.
# 200 -> pad to 208 -> 104 -> 52 -> 26 -> 13
PAD_TOP = 4
PAD_BOTTOM = 4

# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

CHECKPOINT_EVERY = 10
MAP_SAVE_EVERY = 5
MAP_VOLUME_INDEX = 0
MAP_BATCH_SIZE = 16
SAVE_INITIAL_MAP = False

MODEL_ROOT = Path("./models_enhancement")
LATEST_MODEL_DIR = MODEL_ROOT / "latest"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
FINAL_MODEL_DIR = MODEL_ROOT / "final"

OUTPUT_ROOT = Path("./outputs")
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

    # Fixed input shape -> cuDNN can benchmark and cache a fast algorithm.
    torch.backends.cudnn.benchmark = True

    # RTX 30-series supports TF32. AMP is still used for the main forward/backward.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Good default for recent PyTorch versions.
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
# Data loading utilities
# ============================================================


def extract_volume_key(file_path: Path) -> str:
    return re.sub(
        r"_(LOW|HIGH)$",
        "",
        file_path.stem,
        flags=re.IGNORECASE,
    )



def get_bin_files(directory: Path) -> list[Path]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.bin"))

    if not files:
        raise RuntimeError(f"No .bin files found in: {directory}")

    return files



def validate_bin_file(
    file_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype = DATA_DTYPE,
    offset: int = HEADER_SIZE,
) -> None:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dtype = np.dtype(dtype)
    expected_elements = int(np.prod(shape))
    expected_file_size = offset + expected_elements * dtype.itemsize
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

        validate_bin_file(low_path, ADJACENT_LOW_SHAPE)
        validate_bin_file(high_path, HIGH_VOLUME_SHAPE)

        pairs.append(
            {
                "volume_key": key,
                "low_path": low_path,
                "high_path": high_path,
            }
        )

    if not pairs:
        raise RuntimeError("No valid LOW-HIGH pairs found.")

    return pairs


# ============================================================
# Dataset
# ============================================================


class PAMAdjacentXZDataset(Dataset):
    """
    Full indexed PAM dataset.

    LOW file:
        [Y, C, X, Z] = [200, 3, 200, 512]

    HIGH file:
        [X, Y, Z] = [200, 200, 512]

    One sample:
        input  [3, 200, 512]
        target [1, 200, 512]

    The dataset still contains all 26,000 samples.
    Epoch-wise subset selection is handled by EpochRandomSliceSampler.
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
        self.samples_per_volume = Y_SIZE
        self.total_samples = self.num_volumes * self.samples_per_volume

        # Lazy per-process memmap cache.
        self._low_memmaps: dict[int, np.memmap] = {}
        self._high_memmaps: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.total_samples

    def _get_low_memmap(self, volume_index: int) -> np.memmap:
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

    def _get_high_memmap(self, volume_index: int) -> np.memmap:
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Dataset index out of range: {index}. "
                f"Valid range: 0 ~ {len(self) - 1}"
            )

        volume_index, y_index = divmod(index, self.samples_per_volume)
        pair = self.pairs[volume_index]

        low_volume = self._get_low_memmap(volume_index)
        high_volume = self._get_high_memmap(volume_index)

        # LOW [Y, C, X, Z] -> [C, X, Z]
        input_array = np.array(
            low_volume[y_index, :, :, :],
            dtype=np.float32,
            copy=True,
        )

        # HIGH [X, Y, Z] -> [X, Z] -> [1, X, Z]
        target_array = np.array(
            high_volume[:, y_index, :],
            dtype=np.float32,
            copy=True,
        )
        target_array = np.expand_dims(target_array, axis=0)

        if input_array.shape != INPUT_SAMPLE_SHAPE:
            raise ValueError(
                "\nInput shape mismatch\n"
                f"Volume key    : {pair['volume_key']}\n"
                f"Y index       : {y_index}\n"
                f"Actual shape  : {input_array.shape}\n"
                f"Expected shape: {INPUT_SAMPLE_SHAPE}"
            )

        if target_array.shape != TARGET_SAMPLE_SHAPE:
            raise ValueError(
                "\nTarget shape mismatch\n"
                f"Volume key    : {pair['volume_key']}\n"
                f"Y index       : {y_index}\n"
                f"Actual shape  : {target_array.shape}\n"
                f"Expected shape: {TARGET_SAMPLE_SHAPE}"
            )

        return {
            "input": torch.from_numpy(input_array),
            "target": torch.from_numpy(target_array),
            "volume_key": pair["volume_key"],
            "volume_index": volume_index,
            "y_index": y_index,
        }

    def close(self) -> None:
        self._low_memmaps.clear()
        self._high_memmaps.clear()

    def __getstate__(self) -> dict[str, Any]:
        # Open memmaps must not be serialized to worker processes.
        state = self.__dict__.copy()
        state["_low_memmaps"] = {}
        state["_high_memmaps"] = {}
        return state


# ============================================================
# Epoch-wise random slice sampler
# ============================================================


class EpochRandomSliceSampler(Sampler[int]):
    """
    For every epoch:
      1. Shuffle volume order.
      2. Sample N unique Y indices from each volume.
      3. Sort Y indices inside each volume for better I/O locality.

    Example with 130 volumes and N=32:
        130 x 32 = 4,160 samples / epoch.

    The full Dataset remains unchanged, so MAP/inference can still use all
    200 Y slices of a volume.
    """

    def __init__(
        self,
        dataset: PAMAdjacentXZDataset,
        slices_per_volume: int,
        seed: int = 42,
    ) -> None:
        if slices_per_volume <= 0:
            raise ValueError("slices_per_volume must be > 0")

        if slices_per_volume > dataset.samples_per_volume:
            raise ValueError(
                f"slices_per_volume={slices_per_volume} exceeds "
                f"available Y slices={dataset.samples_per_volume}"
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
            y_indices = rng.choice(
                self.dataset.samples_per_volume,
                size=self.slices_per_volume,
                replace=False,
            )

            # Better sequential disk/page-cache behavior within each volume.
            y_indices.sort()

            base_index = int(volume_index) * self.dataset.samples_per_volume
            global_indices.extend(
                base_index + int(y_index)
                for y_index in y_indices
            )

        return iter(global_indices)


# ============================================================
# Model
# ============================================================


class DoubleConv(nn.Module):
    """Conv -> ReLU -> Conv -> ReLU. No BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VanillaUNet(nn.Module):
    """
    PAM 3-adjacent X-Z enhancement U-Net.

    Input:
        [B, 3, 200, 512]

    Channel progression:
        3 -> 16 -> 32 -> 64 -> 128 -> 256
          -> 128 -> 64 -> 32 -> 16 -> 1

    Output:
        [B, 1, 200, 512]

    Final activation:
        None (linear output)
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        out_channels: int = OUTPUT_CHANNELS,
        base_channels: int = BASE_CHANNELS,
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
                f"bottleneck={c5}, MAX_CHANNELS={MAX_CHANNELS}"
            )

        self.encoder1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(c4, c5)

        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(c4 + c4, c4)

        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(c3 + c3, c3)

        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(c2 + c2, c2)

        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(c1 + c1, c1)

        # Linear output: no Sigmoid, no Tanh, no ReLU.
        self.output_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 3, 200, 512] -> [B, 3, 208, 512]
        x = F.pad(
            x,
            pad=(0, 0, PAD_TOP, PAD_BOTTOM),
            mode="reflect",
        )

        e1 = self.encoder1(x)                # [B, 16, 208, 512]
        e2 = self.encoder2(self.pool1(e1))   # [B, 32, 104, 256]
        e3 = self.encoder3(self.pool2(e2))   # [B, 64, 52, 128]
        e4 = self.encoder4(self.pool3(e3))   # [B, 128, 26, 64]

        bottleneck = self.bottleneck(self.pool4(e4))  # [B, 256, 13, 32]

        d4 = self.upconv4(bottleneck)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        output = self.output_conv(d1)  # [B, 1, 208, 512]
        output = output[:, :, PAD_TOP:-PAD_BOTTOM, :]  # [B, 1, 200, 512]

        return output


# ============================================================
# DataLoader creation
# ============================================================



def create_train_loader() -> tuple[
    PAMAdjacentXZDataset,
    EpochRandomSliceSampler,
    DataLoader,
]:
    train_dataset = PAMAdjacentXZDataset(
        low_dir=TRAIN_LOW_DIR,
        high_dir=TRAIN_HIGH_DIR,
    )

    train_sampler = EpochRandomSliceSampler(
        dataset=train_dataset,
        slices_per_volume=SLICES_PER_VOLUME_PER_EPOCH,
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
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(**loader_kwargs)

    return train_dataset, train_sampler, train_loader


# ============================================================
# Verification / helpers
# ============================================================



def count_trainable_parameters(model: nn.Module) -> int:
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
    batch = next(iter(train_loader))

    inputs = batch["input"].to(
        device=device,
        non_blocking=True,
    )
    targets = batch["target"].to(
        device=device,
        non_blocking=True,
    )

    if USE_CHANNELS_LAST and device.type == "cuda":
        inputs = inputs.contiguous(memory_format=torch.channels_last)

    model.eval()

    with torch.cuda.amp.autocast(enabled=amp_enabled):
        outputs = model(inputs)

    print("\n" + "=" * 70)
    print("MODEL SHAPE VERIFICATION")
    print("=" * 70)
    print(f"Input shape : {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Target shape: {targets.shape}")

    if outputs.shape != targets.shape:
        raise ValueError(
            "\nModel output and target shape mismatch.\n"
            f"Output: {outputs.shape}\n"
            f"Target: {targets.shape}"
        )

    print("\nModel output shape is valid.")



def print_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\nCUDA Memory [{label}]")
    print(f"  Allocated : {allocated:.2f} GB")
    print(f"  Reserved  : {reserved:.2f} GB")
    print(f"  Peak      : {peak:.2f} GB")


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

    # Keep accumulation on GPU; avoid loss.item() every iteration.
    running_loss = torch.zeros((), device=device, dtype=torch.float64)
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
        inputs = batch["input"].to(
            device=device,
            non_blocking=True,
        )
        targets = batch["target"].to(
            device=device,
            non_blocking=True,
        )

        if USE_CHANNELS_LAST and device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        current_batch_size = inputs.size(0)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            predictions = model(inputs)

            # Raw linear prediction; no clamp or activation in training.
            raw_loss = criterion(predictions, targets)

            loss_for_backward = (
                raw_loss / GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(loss_for_backward).backward()

        is_accumulation_boundary = (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        )
        is_last_batch = (step + 1 == len(train_loader))

        if is_accumulation_boundary or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += raw_loss.detach().double() * current_batch_size
        total_samples += current_batch_size

        if (
            (step + 1) % LOG_EVERY == 0
            or is_last_batch
        ):
            mean_loss = (running_loss / total_samples).item()
            progress_bar.set_postfix({"L1": f"{mean_loss:.6f}"})

    return (running_loss / total_samples).item()


# ============================================================
# Full-volume MAP visualization
# ============================================================


@torch.no_grad()
def save_epoch_map(
    model: nn.Module,
    train_dataset: PAMAdjacentXZDataset,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    volume_index: int = MAP_VOLUME_INDEX,
) -> None:
    """
    Uses all 200 Y slices of one fixed Train volume.

    200 X-Z predictions
        -> [Y, 1, X, Z]
        -> [Y, X, Z]
        -> transpose to [X, Y, Z]
        -> max over Z
        -> X-Y MAP [200, 200]
    """

    if volume_index < 0 or volume_index >= train_dataset.num_volumes:
        raise IndexError(
            f"Invalid MAP volume index: {volume_index}. "
            f"Valid range: 0 ~ {train_dataset.num_volumes - 1}"
        )

    model.eval()

    start_index = volume_index * train_dataset.samples_per_volume
    end_index = start_index + train_dataset.samples_per_volume

    volume_subset = Subset(
        train_dataset,
        list(range(start_index, end_index)),
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

        if USE_CHANNELS_LAST and device.type == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            predictions = model(inputs)

        # Visualization only.
        predictions_for_map = predictions.clamp_min(0.0)

        # Center LOW channel: [B, 3, X, Z] -> [B, 1, X, Z]
        current_low = inputs[:, 1:2, :, :]

        low_slices.append(current_low.detach().float().cpu())
        prediction_slices.append(
            predictions_for_map.detach().float().cpu()
        )
        target_slices.append(targets.detach().float().cpu())

        if volume_key is None:
            volume_key = batch["volume_key"][0]

    low_slices_tensor = torch.cat(low_slices, dim=0)
    prediction_slices_tensor = torch.cat(prediction_slices, dim=0)
    target_slices_tensor = torch.cat(target_slices, dim=0)

    expected_shape = (
        train_dataset.samples_per_volume,
        1,
        X_SIZE,
        Z_SIZE,
    )

    for label, tensor in (
        ("LOW", low_slices_tensor),
        ("Prediction", prediction_slices_tensor),
        ("Target", target_slices_tensor),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"\n{label} reconstruction shape mismatch.\n"
                f"Actual   : {tuple(tensor.shape)}\n"
                f"Expected : {expected_shape}"
            )

    # [Y, 1, X, Z] -> [Y, X, Z]
    low_yxz = low_slices_tensor[:, 0].numpy()
    pred_yxz = prediction_slices_tensor[:, 0].numpy()
    target_yxz = target_slices_tensor[:, 0].numpy()

    # [Y, X, Z] -> [X, Y, Z]
    low_volume = np.transpose(low_yxz, axes=(1, 0, 2))
    pred_volume = np.transpose(pred_yxz, axes=(1, 0, 2))
    target_volume = np.transpose(target_yxz, axes=(1, 0, 2))

    # X-Y MAP via maximum projection over Z.
    low_map = np.max(low_volume, axis=2)
    pred_map = np.max(pred_volume, axis=2)
    target_map = np.max(target_volume, axis=2)

    absolute_error_map = np.abs(pred_map - target_map)

    display_vmin = 0.0
    display_vmax = float(np.percentile(target_map, 99.5))

    if not np.isfinite(display_vmax) or display_vmax <= display_vmin:
        display_vmax = float(target_map.max())

    if display_vmax <= display_vmin:
        display_vmax = 1.0

    error_vmax = float(np.percentile(absolute_error_map, 99.5))

    if not np.isfinite(error_vmax) or error_vmax <= 0:
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
    figure.colorbar(image_0, ax=axes[0], fraction=0.046, pad=0.04)

    image_1 = axes[1].imshow(
        pred_map,
        cmap="hot",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[1].set_title(f"Prediction MAP\nEpoch {epoch}")
    axes[1].axis("off")
    figure.colorbar(image_1, ax=axes[1], fraction=0.046, pad=0.04)

    image_2 = axes[2].imshow(
        target_map,
        cmap="hot",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes[2].set_title("HIGH MAP")
    axes[2].axis("off")
    figure.colorbar(image_2, ax=axes[2], fraction=0.046, pad=0.04)

    image_3 = axes[3].imshow(
        absolute_error_map,
        cmap="viridis",
        vmin=0.0,
        vmax=error_vmax,
    )
    axes[3].set_title("Absolute Error")
    axes[3].axis("off")
    figure.colorbar(image_3, ax=axes[3], fraction=0.046, pad=0.04)

    figure.suptitle(
        f"Vanilla U-Net Train MAP | {volume_key} | Epoch {epoch}",
        fontsize=14,
    )

    figure.tight_layout()

    output_path = TRAIN_MAP_DIR / f"epoch_{epoch:03d}_map.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved Train MAP: {output_path}")

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
    path: Path,
) -> None:
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_loss": train_loss,
        "model_name": "VanillaUNet",
        "input_channels": INPUT_CHANNELS,
        "output_channels": OUTPUT_CHANNELS,
        "base_channels": BASE_CHANNELS,
        "max_channels": MAX_CHANNELS,
        "physical_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": (
            BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "slices_per_volume_per_epoch": SLICES_PER_VOLUME_PER_EPOCH,
        "samples_per_epoch": (
            SLICES_PER_VOLUME_PER_EPOCH
            * 130  # informational only; actual value is printed at runtime
        ),
        "use_amp": USE_AMP,
        "use_channels_last": USE_CHANNELS_LAST,
        "final_output_activation": "None (Linear)",
        "loss": "L1Loss",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "map_save_every": MAP_SAVE_EVERY,
    }

    torch.save(checkpoint_data, path)



def save_training_history(history: list[dict[str, float]]) -> None:
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



def save_loss_curve(history: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in history]
    losses = [row["train_l1_loss"] for row in history]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Train L1 Loss")
    plt.title("Vanilla U-Net Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Full training
# ============================================================



def create_optimizer(
    model: nn.Module,
    device: torch.device,
) -> torch.optim.Optimizer:
    """
    Use fused Adam when supported by the local PyTorch/CUDA build;
    otherwise fall back to standard Adam.
    """

    base_kwargs = {
        "params": model.parameters(),
        "lr": LEARNING_RATE,
        "betas": (0.9, 0.999),
    }

    if device.type == "cuda":
        try:
            optimizer = torch.optim.Adam(**base_kwargs, fused=True)
            print("  Adam implementation    : fused")
            return optimizer
        except (TypeError, RuntimeError):
            pass

    print("  Adam implementation    : standard")
    return torch.optim.Adam(**base_kwargs)



def train() -> None:
    set_seed(SEED)
    configure_cuda()
    create_output_directories()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = USE_AMP and device.type == "cuda"

    print("\n" + "=" * 70)
    print("PAM 3-ADJACENT X-Z VANILLA U-NET TRAINING")
    print("EPOCH-WISE RANDOM Y-SLICE SAMPLING")
    print("=" * 70)
    print(f"Device                  : {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        )
        print(f"GPU                     : {gpu_name}")
        print(f"Total GPU memory        : {total_gpu_memory:.2f} GB")

    train_dataset, train_sampler, train_loader = create_train_loader()

    samples_per_epoch = len(train_sampler)

    print("\nData")
    print(f"  Train volumes         : {train_dataset.num_volumes}")
    print(f"  Full train samples    : {len(train_dataset):,}")
    print(
        f"  Slices/volume/epoch   : "
        f"{SLICES_PER_VOLUME_PER_EPOCH} / {train_dataset.samples_per_volume}"
    )
    print(f"  Samples this epoch    : {samples_per_epoch:,}")
    print(f"  Physical batch size   : {BATCH_SIZE}")
    print(f"  Batches/epoch         : {len(train_loader):,}")
    print(f"  Num workers           : {NUM_WORKERS}")
    print(f"  Persistent workers    : {PERSISTENT_WORKERS}")
    print(f"  Prefetch factor       : {PREFETCH_FACTOR if NUM_WORKERS > 0 else 'N/A'}")

    model = VanillaUNet(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        base_channels=BASE_CHANNELS,
    ).to(device)

    if USE_CHANNELS_LAST and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    parameter_count = count_trainable_parameters(model)

    print("\nModel")
    print("  Architecture           : Vanilla U-Net")
    print("  Channels               : 3-16-32-64-128-256-128-64-32-16-1")
    print(f"  Parameters             : {parameter_count:,}")
    print("  BatchNorm              : None")
    print("  Final output activation: None (Linear)")
    print(f"  Channels last          : {USE_CHANNELS_LAST}")

    criterion = nn.L1Loss()

    print("\nOptimizer")
    optimizer = create_optimizer(model, device)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print("\nTraining configuration")
    print(f"  Epochs                 : {NUM_EPOCHS}")
    print(f"  Learning rate          : {LEARNING_RATE}")
    print("  Loss                   : L1 Loss")
    print("  Optimizer              : Adam")
    print(f"  AMP                    : {amp_enabled}")
    print("  Validation             : None")
    print("  Test                   : Not used")
    print(f"  MAP save every         : {MAP_SAVE_EVERY} epochs")
    print(f"  MAP volume index       : {MAP_VOLUME_INDEX}")

    # Epoch 0 sampling for shape verification only.
    train_sampler.set_epoch(0)
    verify_model_shape(
        model=model,
        train_loader=train_loader,
        device=device,
        amp_enabled=amp_enabled,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print_cuda_memory("Before Training")

    if SAVE_INITIAL_MAP:
        print("\nGenerating initial Epoch 0 MAP...")
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

    for epoch in range(1, NUM_EPOCHS + 1):
        # New deterministic random Y subset for every epoch.
        train_sampler.set_epoch(epoch)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        epoch_start_time = time.perf_counter()

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

        epoch_time = time.perf_counter() - epoch_start_time

        if torch.cuda.is_available():
            peak_gpu_memory_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
        else:
            peak_gpu_memory_gb = 0.0

        history.append(
            {
                "epoch": epoch,
                "train_l1_loss": train_loss,
                "epoch_time_seconds": epoch_time,
                "peak_gpu_memory_gb": peak_gpu_memory_gb,
                "samples_this_epoch": samples_per_epoch,
            }
        )

        print(f"\nEpoch {epoch:03d}/{NUM_EPOCHS:03d}")
        print(f"Train L1 Loss   : {train_loss:.8f}")
        print(f"Epoch Time      : {epoch_time:.2f} sec")
        print(f"Peak GPU Memory : {peak_gpu_memory_gb:.2f} GB")
        print(f"Samples trained : {samples_per_epoch:,}")

        # Full 200-slice MAP inference every 5 epochs.
        if epoch % MAP_SAVE_EVERY == 0:
            print(f"\nGenerating full-volume MAP for Epoch {epoch}...")
            save_epoch_map(
                model=model,
                train_dataset=train_dataset,
                device=device,
                epoch=epoch,
                amp_enabled=amp_enabled,
                volume_index=MAP_VOLUME_INDEX,
            )

        # Latest checkpoint every epoch.
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            train_loss=train_loss,
            path=LATEST_MODEL_PATH,
        )

        # Periodic checkpoint every 10 epochs.
        if epoch % CHECKPOINT_EVERY == 0:
            checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                train_loss=train_loss,
                path=checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        save_training_history(history)
        save_loss_curve(history)

    final_train_loss = history[-1]["train_l1_loss"]

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=NUM_EPOCHS,
        train_loss=final_train_loss,
        path=FINAL_MODEL_PATH,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal model:\n  {FINAL_MODEL_PATH}")
    print(f"\nLatest model:\n  {LATEST_MODEL_PATH}")
    print(f"\nTraining MAP images:\n  {TRAIN_MAP_DIR}")
    print(f"\nTraining history:\n  {HISTORY_CSV_PATH}")
    print(f"\nLoss curve:\n  {LOSS_CURVE_PATH}")

    train_dataset.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    train()
