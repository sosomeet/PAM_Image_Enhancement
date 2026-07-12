from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("./data")

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

TRAIN_LOW_DIR = DATA_ROOT / "3d_Train" / "LOW"
TRAIN_HIGH_DIR = DATA_ROOT / "norm_Train" / "HIGH"

# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

TEST_LOW_DIR = DATA_ROOT / "3d_Test" / "LOW"
TEST_HIGH_DIR = DATA_ROOT / "norm_Test" / "HIGH"


# ============================================================
# Data specification
# ============================================================

HEADER_SIZE = 48
DATA_DTYPE = np.float32

# Original normalized HIGH volume
#
# [X, Y, Z]
# [200, 200, 512]
#
HIGH_VOLUME_SHAPE = (
    200,  # X
    200,  # Y
    512,  # Z
)

# Generated 3-adjacent LOW data
#
# [Y, C, X, Z]
# [200, 3, 200, 512]
#
ADJACENT_LOW_SHAPE = (
    200,  # Y samples
    3,    # previous, current, next
    200,  # X
    512,  # Z
)

X_SIZE = HIGH_VOLUME_SHAPE[0]   # 200
Y_SIZE = HIGH_VOLUME_SHAPE[1]   # 200
Z_SIZE = HIGH_VOLUME_SHAPE[2]   # 512

NUM_CHANNELS = 3

INPUT_SAMPLE_SHAPE = (
    NUM_CHANNELS,
    X_SIZE,
    Z_SIZE,
)

TARGET_SAMPLE_SHAPE = (
    1,
    X_SIZE,
    Z_SIZE,
)


# ============================================================
# DataLoader configuration
# ============================================================

TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1

# Current server/environment setting
NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 1. Extract volume key
# ============================================================

def extract_volume_key(
    file_path: Path,
) -> str:
    """
    Extract a common volume identifier from LOW/HIGH filenames.

    Examples
    --------
    Train_000_LOW.bin
        -> Train_000

    Train_000_HIGH.bin
        -> Train_000

    Test_003_LOW.bin
        -> Test_003

    Test_003_HIGH.bin
        -> Test_003
    """

    stem = file_path.stem

    key = re.sub(
        r"_(LOW|HIGH)$",
        "",
        stem,
        flags=re.IGNORECASE,
    )

    return key


# ============================================================
# 2. Find .bin files
# ============================================================

def get_bin_files(
    directory: Path,
) -> list[Path]:
    """
    Find all .bin files and sort them.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = sorted(
        directory.glob("*.bin")
    )

    if not files:
        raise RuntimeError(
            f"No .bin files found in: {directory}"
        )

    return files


# ============================================================
# 3. Validate file size
# ============================================================

def validate_bin_file(
    file_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype = DATA_DTYPE,
    offset: int = HEADER_SIZE,
) -> None:
    """
    Validate binary file size against:

        header
        +
        payload shape
        ×
        dtype size
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    dtype = np.dtype(dtype)

    expected_elements = int(
        np.prod(shape)
    )

    expected_payload_bytes = (
        expected_elements
        * dtype.itemsize
    )

    expected_file_size = (
        offset
        + expected_payload_bytes
    )

    actual_file_size = (
        file_path.stat().st_size
    )

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


# ============================================================
# 4. Build LOW-HIGH file pairs
# ============================================================

def build_file_pairs(
    low_dir: Path,
    high_dir: Path,
) -> list[dict[str, Any]]:
    """
    Match LOW and HIGH files using their common volume key.

    Example:

        Train_000_LOW.bin
            ↕
        Train_000_HIGH.bin
    """

    low_files = get_bin_files(
        low_dir
    )

    high_files = get_bin_files(
        high_dir
    )

    low_map = {
        extract_volume_key(file_path): file_path
        for file_path in low_files
    }

    high_map = {
        extract_volume_key(file_path): file_path
        for file_path in high_files
    }

    low_keys = set(
        low_map.keys()
    )

    high_keys = set(
        high_map.keys()
    )

    # --------------------------------------------------------
    # Check unmatched files
    # --------------------------------------------------------

    missing_high = sorted(
        low_keys - high_keys
    )

    missing_low = sorted(
        high_keys - low_keys
    )

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

    # --------------------------------------------------------
    # Build pairs
    # --------------------------------------------------------

    common_keys = sorted(
        low_keys & high_keys
    )

    pairs = []

    for key in common_keys:

        low_path = low_map[key]
        high_path = high_map[key]

        # Validate generated LOW file
        validate_bin_file(
            file_path=low_path,
            shape=ADJACENT_LOW_SHAPE,
        )

        # Validate normalized HIGH file
        validate_bin_file(
            file_path=high_path,
            shape=HIGH_VOLUME_SHAPE,
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
# 5. PAM Dataset
# ============================================================

class PAMAdjacentXZDataset(Dataset):
    """
    PAM image-enhancement Dataset.

    LOW input
    ---------
    File shape:

        [Y, C, X, Z]
        [200, 3, 200, 512]

    One input sample:

        [C, X, Z]
        [3, 200, 512]

    HIGH target
    -----------
    File shape:

        [X, Y, Z]
        [200, 200, 512]

    One target sample:

        HIGH[:, y, :]

        [X, Z]
        [200, 512]

    Returned target shape:

        [1, X, Z]
        [1, 200, 512]
    """

    def __init__(
        self,
        low_dir: Path,
        high_dir: Path,
    ) -> None:

        super().__init__()

        self.low_dir = Path(
            low_dir
        )

        self.high_dir = Path(
            high_dir
        )

        # ----------------------------------------------------
        # Build and validate LOW-HIGH volume pairs
        # ----------------------------------------------------

        self.pairs = build_file_pairs(
            low_dir=self.low_dir,
            high_dir=self.high_dir,
        )

        # ----------------------------------------------------
        # Number of volumes
        # ----------------------------------------------------

        self.num_volumes = len(
            self.pairs
        )

        # ----------------------------------------------------
        # 200 Y samples per volume
        # ----------------------------------------------------

        self.samples_per_volume = (
            Y_SIZE
        )

        self.total_samples = (
            self.num_volumes
            * self.samples_per_volume
        )

        # ----------------------------------------------------
        # Lazy memmap cache
        #
        # With NUM_WORKERS = 0, the same opened mappings can be
        # reused during training instead of reopening files
        # for every __getitem__ call.
        # ----------------------------------------------------

        self._low_memmaps: dict[
            int,
            np.memmap,
        ] = {}

        self._high_memmaps: dict[
            int,
            np.memmap,
        ] = {}

    def __len__(self) -> int:
        """
        Total number of X-Z samples.

        num_volumes × 200
        """

        return self.total_samples

    def _get_low_memmap(
        self,
        volume_index: int,
    ) -> np.memmap:
        """
        Lazily open and cache one adjacent LOW file.
        """

        if volume_index not in self._low_memmaps:

            low_path = self.pairs[
                volume_index
            ]["low_path"]

            low_memmap = np.memmap(
                filename=low_path,
                dtype=DATA_DTYPE,
                mode="r",
                offset=HEADER_SIZE,
                shape=ADJACENT_LOW_SHAPE,
                order="C",
            )

            self._low_memmaps[
                volume_index
            ] = low_memmap

        return self._low_memmaps[
            volume_index
        ]

    def _get_high_memmap(
        self,
        volume_index: int,
    ) -> np.memmap:
        """
        Lazily open and cache one normalized HIGH volume.
        """

        if volume_index not in self._high_memmaps:

            high_path = self.pairs[
                volume_index
            ]["high_path"]

            high_memmap = np.memmap(
                filename=high_path,
                dtype=DATA_DTYPE,
                mode="r",
                offset=HEADER_SIZE,
                shape=HIGH_VOLUME_SHAPE,
                order="C",
            )

            self._high_memmaps[
                volume_index
            ] = high_memmap

        return self._high_memmaps[
            volume_index
        ]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Return one training/testing sample.

        Returns
        -------
        {
            "input":
                Tensor [3, 200, 512],

            "target":
                Tensor [1, 200, 512],

            "volume_key":
                str,

            "volume_index":
                int,

            "y_index":
                int,
        }
        """

        if index < 0 or index >= len(self):
            raise IndexError(
                f"Dataset index out of range: {index}. "
                f"Valid range: 0 ~ {len(self) - 1}"
            )

        # ----------------------------------------------------
        # Global sample index
        # ->
        # volume index + Y position
        #
        # Example:
        #
        # index = 0
        #   -> volume 0, y=0
        #
        # index = 199
        #   -> volume 0, y=199
        #
        # index = 200
        #   -> volume 1, y=0
        # ----------------------------------------------------

        volume_index, y_index = divmod(
            index,
            self.samples_per_volume,
        )

        pair = self.pairs[
            volume_index
        ]

        # ----------------------------------------------------
        # Open memmaps
        # ----------------------------------------------------

        low_volume = self._get_low_memmap(
            volume_index
        )

        high_volume = self._get_high_memmap(
            volume_index
        )

        # ----------------------------------------------------
        # Input
        #
        # LOW:
        # [Y, C, X, Z]
        #
        # low_volume[y_index]
        # ->
        # [C, X, Z]
        # [3, 200, 512]
        # ----------------------------------------------------

        input_array = np.array(
            low_volume[
                y_index,
                :,
                :,
                :,
            ],
            dtype=np.float32,
            copy=True,
        )

        # ----------------------------------------------------
        # Target
        #
        # HIGH:
        # [X, Y, Z]
        #
        # high_volume[:, y_index, :]
        # ->
        # [X, Z]
        #
        # Add channel dimension:
        # [1, X, Z]
        # ----------------------------------------------------

        target_array = np.array(
            high_volume[
                :,
                y_index,
                :,
            ],
            dtype=np.float32,
            copy=True,
        )

        target_array = np.expand_dims(
            target_array,
            axis=0,
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(
            input_array
        ).float()

        target_tensor = torch.from_numpy(
            target_array
        ).float()

        return {
            "input": input_tensor,
            "target": target_tensor,
            "volume_key": pair[
                "volume_key"
            ],
            "volume_index": volume_index,
            "y_index": y_index,
        }

    def close(self) -> None:
        """
        Clear cached memmap objects.
        """

        self._low_memmaps.clear()
        self._high_memmaps.clear()

    def __getstate__(
        self,
    ) -> dict[str, Any]:
        """
        Prevent open memmaps from being serialized if
        DataLoader multiprocessing is used later.
        """

        state = self.__dict__.copy()

        state["_low_memmaps"] = {}
        state["_high_memmaps"] = {}

        return state


# ============================================================
# 6. Verify one Dataset sample
# ============================================================

def verify_dataset_sample(
    dataset: PAMAdjacentXZDataset,
    index: int = 0,
) -> None:
    """
    Verify shape, dtype, range, NaN and Inf.
    """

    sample = dataset[index]

    input_tensor = sample["input"]
    target_tensor = sample["target"]

    print("\n" + "=" * 70)
    print("PAM DATASET SAMPLE VERIFICATION")
    print("=" * 70)

    print(
        f"Dataset index : {index}"
    )

    print(
        f"Volume key    : "
        f"{sample['volume_key']}"
    )

    print(
        f"Volume index  : "
        f"{sample['volume_index']}"
    )

    print(
        f"Y index       : "
        f"{sample['y_index']}"
    )

    print("\nInput")

    print(
        f"  Shape : {input_tensor.shape}"
    )

    print(
        f"  Dtype : {input_tensor.dtype}"
    )

    print(
        f"  Min   : "
        f"{input_tensor.min().item():.6f}"
    )

    print(
        f"  Max   : "
        f"{input_tensor.max().item():.6f}"
    )

    print(
        f"  Mean  : "
        f"{input_tensor.mean().item():.6f}"
    )

    print(
        f"  NaN   : "
        f"{torch.isnan(input_tensor).any().item()}"
    )

    print(
        f"  Inf   : "
        f"{torch.isinf(input_tensor).any().item()}"
    )

    print("\nTarget")

    print(
        f"  Shape : {target_tensor.shape}"
    )

    print(
        f"  Dtype : {target_tensor.dtype}"
    )

    print(
        f"  Min   : "
        f"{target_tensor.min().item():.6f}"
    )

    print(
        f"  Max   : "
        f"{target_tensor.max().item():.6f}"
    )

    print(
        f"  Mean  : "
        f"{target_tensor.mean().item():.6f}"
    )

    print(
        f"  NaN   : "
        f"{torch.isnan(target_tensor).any().item()}"
    )

    print(
        f"  Inf   : "
        f"{torch.isinf(target_tensor).any().item()}"
    )


# ============================================================
# 7. Verify LOW center channel
# ============================================================

def verify_center_channel(
    dataset: PAMAdjacentXZDataset,
    index: int = 0,
) -> None:
    """
    Verify the Dataset input center channel.

    Input channels:

        channel 0 = previous X-Z LOW
        channel 1 = current X-Z LOW
        channel 2 = next X-Z LOW

    This function reports differences between adjacent channels.
    """

    sample = dataset[index]

    input_tensor = sample["input"]

    previous_slice = input_tensor[0]
    current_slice = input_tensor[1]
    next_slice = input_tensor[2]

    previous_difference = torch.mean(
        torch.abs(
            previous_slice
            - current_slice
        )
    )

    next_difference = torch.mean(
        torch.abs(
            current_slice
            - next_slice
        )
    )

    print("\n" + "=" * 70)
    print("3-ADJACENT CHANNEL VERIFICATION")
    print("=" * 70)

    print(
        f"Volume key: "
        f"{sample['volume_key']}"
    )

    print(
        f"Y index   : "
        f"{sample['y_index']}"
    )

    print(
        "\nMean absolute difference"
    )

    print(
        "  Previous vs Current : "
        f"{previous_difference.item():.8f}"
    )

    print(
        "  Current vs Next     : "
        f"{next_difference.item():.8f}"
    )


# ============================================================
# 8. Create DataLoaders
# ============================================================

def create_dataloaders() -> tuple[
    PAMAdjacentXZDataset,
    PAMAdjacentXZDataset,
    DataLoader,
    DataLoader,
]:
    """
    Create Train/Test Dataset and DataLoader.
    """

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = PAMAdjacentXZDataset(
        low_dir=TRAIN_LOW_DIR,
        high_dir=TRAIN_HIGH_DIR,
    )

    test_dataset = PAMAdjacentXZDataset(
        low_dir=TEST_LOW_DIR,
        high_dir=TEST_HIGH_DIR,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
    )


# ============================================================
# 9. Verify one batch
# ============================================================

def verify_batch(
    loader: DataLoader,
    label: str,
) -> None:
    """
    Read and inspect one DataLoader batch.
    """

    batch = next(
        iter(loader)
    )

    inputs = batch["input"]
    targets = batch["target"]

    print("\n" + "=" * 70)
    print(f"{label} BATCH VERIFICATION")
    print("=" * 70)

    print(
        f"Input shape : {inputs.shape}"
    )

    print(
        f"Target shape: {targets.shape}"
    )

    print(
        f"Input dtype : {inputs.dtype}"
    )

    print(
        f"Target dtype: {targets.dtype}"
    )

    print(
        f"Volume keys : "
        f"{batch['volume_key']}"
    )

    print(
        f"Y indices   : "
        f"{batch['y_index']}"
    )

    print("\nInput statistics")

    print(
        f"  Min  : "
        f"{inputs.min().item():.6f}"
    )

    print(
        f"  Max  : "
        f"{inputs.max().item():.6f}"
    )

    print(
        f"  Mean : "
        f"{inputs.mean().item():.6f}"
    )

    print("\nTarget statistics")

    print(
        f"  Min  : "
        f"{targets.min().item():.6f}"
    )

    print(
        f"  Max  : "
        f"{targets.max().item():.6f}"
    )

    print(
        f"  Mean : "
        f"{targets.mean().item():.6f}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("PAM 3-ADJACENT X-Z DATA LOADER")
    print("=" * 70)

    print("\nData specification")

    print(
        f"  LOW file shape   : "
        f"{ADJACENT_LOW_SHAPE}"
    )

    print(
        f"  HIGH file shape  : "
        f"{HIGH_VOLUME_SHAPE}"
    )

    print(
        f"  Input sample     : "
        f"{INPUT_SAMPLE_SHAPE}"
    )

    print(
        f"  Target sample    : "
        f"{TARGET_SAMPLE_SHAPE}"
    )

    print(
        f"  Header size      : "
        f"{HEADER_SIZE} bytes"
    )

    print(
        f"  Dtype            : "
        f"{np.dtype(DATA_DTYPE)}"
    )

    print(
        f"  Train batch size : "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"  Test batch size  : "
        f"{TEST_BATCH_SIZE}"
    )

    print(
        f"  Num workers      : "
        f"{NUM_WORKERS}"
    )

    # --------------------------------------------------------
    # Create Dataset / DataLoader
    # --------------------------------------------------------

    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
    ) = create_dataloaders()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Train volumes : "
        f"{train_dataset.num_volumes}"
    )

    print(
        f"Train samples : "
        f"{len(train_dataset)}"
    )

    print(
        f"Test volumes  : "
        f"{test_dataset.num_volumes}"
    )

    print(
        f"Test samples  : "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Verify Train samples
    # --------------------------------------------------------

    verify_dataset_sample(
        dataset=train_dataset,
        index=0,
    )

    verify_dataset_sample(
        dataset=train_dataset,
        index=100,
    )

    verify_dataset_sample(
        dataset=train_dataset,
        index=199,
    )

    # --------------------------------------------------------
    # Verify adjacent channels
    # --------------------------------------------------------

    verify_center_channel(
        dataset=train_dataset,
        index=100,
    )

    # --------------------------------------------------------
    # Verify batches
    # --------------------------------------------------------

    verify_batch(
        loader=train_loader,
        label="TRAIN",
    )

    verify_batch(
        loader=test_loader,
        label="TEST",
    )

    print("\n" + "=" * 70)
    print("PAM DATA LOADER VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()