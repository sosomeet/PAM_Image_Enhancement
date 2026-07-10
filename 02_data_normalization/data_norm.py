from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("./data")

VOLUME_SHAPE = (200, 200, 512)  # [H, W, D]
DTYPE = np.uint16
OFFSET = 48  # Header size in bytes

EPS = 1e-8


# ============================================================
# 1. Load 3D .bin volume
# ============================================================

def load_bin_volume(
    file_path: Path,
    shape: Tuple[int, int, int] = VOLUME_SHAPE,
    dtype=np.uint16,
    offset: int = OFFSET,
) -> np.ndarray:
    """
    Load a 3D volume from a .bin file.

    Args:
        file_path:
            Path to .bin file.

        shape:
            Volume shape in [H, W, D] format.

        dtype:
            Original data type.

        offset:
            Header offset in bytes.

    Returns:
        volume:
            NumPy array with shape [H, W, D].
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    raw = np.fromfile(
        file_path,
        dtype=dtype,
        offset=offset,
    )

    expected_size = int(np.prod(shape))

    if raw.size != expected_size:
        raise ValueError(
            f"\nData size mismatch: {file_path}\n"
            f"Raw size      : {raw.size}\n"
            f"Expected size : {expected_size}\n"
            f"Expected shape: {shape}"
        )

    volume = raw.reshape(shape)

    return volume


# ============================================================
# 2. 3D volume-wise Min-Max normalization
# ============================================================

def normalize_volume(
    volume: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """
    Apply Min-Max normalization using all voxels
    in one complete 3D volume.

    Input:
        volume: [H, W, D]

    Output:
        normalized volume: [H, W, D], range [0, 1]
    """

    volume = volume.astype(np.float32)

    volume_min = float(volume.min())
    volume_max = float(volume.max())

    if volume_max - volume_min < eps:
        return np.zeros_like(volume, dtype=np.float32)

    normalized = (
        (volume - volume_min)
        / (volume_max - volume_min)
    )

    return normalized.astype(np.float32)


# ============================================================
# 3. Find paired HIGH / LOW files
# ============================================================

def get_paired_files(
    data_root: Path,
    split: str,
):
    """
    Find HIGH / LOW paired .bin files.

    Example:
        Train_000_HIGH.bin
        Train_000_LOW.bin
    """

    split = split.capitalize()

    if split not in ["Train", "Test"]:
        raise ValueError(
            f"split must be 'Train' or 'Test', got: {split}"
        )

    high_dir = data_root / split / "HIGH"
    low_dir = data_root / split / "LOW"

    if not high_dir.exists():
        raise FileNotFoundError(
            f"HIGH directory not found: {high_dir}"
        )

    if not low_dir.exists():
        raise FileNotFoundError(
            f"LOW directory not found: {low_dir}"
        )

    high_files = sorted(high_dir.glob("*.bin"))
    low_files = sorted(low_dir.glob("*.bin"))

    if len(high_files) == 0:
        raise RuntimeError(
            f"No .bin files found in: {high_dir}"
        )

    if len(low_files) == 0:
        raise RuntimeError(
            f"No .bin files found in: {low_dir}"
        )

    if len(high_files) != len(low_files):
        raise ValueError(
            f"\nNumber of HIGH and LOW files does not match.\n"
            f"HIGH: {len(high_files)}\n"
            f"LOW : {len(low_files)}"
        )

    pairs = list(zip(low_files, high_files))

    return pairs


# ============================================================
# 4. PyTorch Dataset
# ============================================================

class PAMVolumeDataset(Dataset):
    """
    Dataset for paired LOW / HIGH 3D PAM volumes.

    Each item:
        LOW  : [1, H, W, D]
        HIGH : [1, H, W, D]

    Each volume is independently normalized
    over the entire 3D volume.
    """

    def __init__(
        self,
        data_root: str = "../data",
        split: str = "Train",
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize

        self.pairs = get_paired_files(
            data_root=self.data_root,
            split=self.split,
        )

        print(
            f"[{self.split}] "
            f"Found {len(self.pairs)} paired volumes."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        low_path, high_path = self.pairs[index]

        # ----------------------------
        # Load raw volumes
        # ----------------------------
        low_volume = load_bin_volume(low_path)
        high_volume = load_bin_volume(high_path)

        # ----------------------------
        # 3D volume normalization
        # ----------------------------
        if self.normalize:
            low_volume = normalize_volume(low_volume)
            high_volume = normalize_volume(high_volume)

        else:
            low_volume = low_volume.astype(np.float32)
            high_volume = high_volume.astype(np.float32)

        # ----------------------------
        # NumPy -> PyTorch Tensor
        # [H, W, D] -> [1, H, W, D]
        # ----------------------------
        low_tensor = torch.from_numpy(
            np.ascontiguousarray(low_volume)
        ).unsqueeze(0)

        high_tensor = torch.from_numpy(
            np.ascontiguousarray(high_volume)
        ).unsqueeze(0)

        sample = {
            "low": low_tensor,
            "high": high_tensor,
            "low_path": str(low_path),
            "high_path": str(high_path),
        }

        return sample


# ============================================================
# 5. Test normalization
# ============================================================

def test_dataset():
    """
    Load one Train sample and print statistics
    before / after normalization.
    """

    dataset = PAMVolumeDataset(
        data_root=DATA_ROOT,
        split="Train",
        normalize=True,
    )

    sample = dataset[0]

    low = sample["low"]
    high = sample["high"]

    print("\n" + "=" * 60)
    print("Sample Information")
    print("=" * 60)

    print(f"LOW file : {sample['low_path']}")
    print(f"HIGH file: {sample['high_path']}")

    print("\nLOW volume")
    print(f"  Shape : {tuple(low.shape)}")
    print(f"  Dtype : {low.dtype}")
    print(f"  Min   : {low.min().item():.6f}")
    print(f"  Max   : {low.max().item():.6f}")
    print(f"  Mean  : {low.mean().item():.6f}")

    print("\nHIGH volume")
    print(f"  Shape : {tuple(high.shape)}")
    print(f"  Dtype : {high.dtype}")
    print(f"  Min   : {high.min().item():.6f}")
    print(f"  Max   : {high.max().item():.6f}")
    print(f"  Mean  : {high.mean().item():.6f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_dataset()