from __future__ import annotations

from pathlib import Path
import json

import numpy as np


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("./data")

# Input normalized LOW volumes
NORM_TRAIN_LOW_DIR = DATA_ROOT / "norm_Train" / "LOW"
NORM_TEST_LOW_DIR = DATA_ROOT / "norm_Test" / "LOW"

# Output 3-adjacent-slice LOW volumes
OUTPUT_TRAIN_LOW_DIR = DATA_ROOT / "3d_Train" / "LOW"
OUTPUT_TEST_LOW_DIR = DATA_ROOT / "3d_Test" / "LOW"

# Normalized PAM volume configuration
VOLUME_SHAPE = (200, 200, 512)  # [H, W, D]

INPUT_DTYPE = np.float32
OUTPUT_DTYPE = np.float32

HEADER_SIZE = 48

# Three adjacent slices:
# previous, current, next
NUM_CHANNELS = 3

# Output payload shape:
# [D, C, H, W]
OUTPUT_SHAPE = (
    VOLUME_SHAPE[2],   # D = 512
    NUM_CHANNELS,      # C = 3
    VOLUME_SHAPE[0],   # H = 200
    VOLUME_SHAPE[1],   # W = 200
)


# ============================================================
# 1. Find .bin files
# ============================================================

def get_bin_files(directory: Path) -> list[Path]:
    """
    Find and sort all .bin files in a directory.
    """

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


# ============================================================
# 2. Read original 48-byte header
# ============================================================

def read_header(
    file_path: Path,
    header_size: int = HEADER_SIZE,
) -> bytes:
    """
    Read the original 48-byte header.
    """

    file_path = Path(file_path)

    with file_path.open("rb") as f:
        header = f.read(header_size)

    if len(header) != header_size:
        raise ValueError(
            f"Header size mismatch.\n"
            f"File     : {file_path}\n"
            f"Expected : {header_size} bytes\n"
            f"Actual   : {len(header)} bytes"
        )

    return header


# ============================================================
# 3. Validate normalized LOW .bin file
# ============================================================

def validate_normalized_file(
    file_path: Path,
    shape: tuple[int, int, int] = VOLUME_SHAPE,
    dtype: np.dtype = INPUT_DTYPE,
    offset: int = HEADER_SIZE,
) -> None:
    """
    Validate normalized LOW .bin file size.

    Expected format:
        [48-byte header]
        +
        [float32 payload with shape H x W x D]
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    dtype = np.dtype(dtype)

    expected_voxels = int(np.prod(shape))

    expected_payload_bytes = (
        expected_voxels
        * dtype.itemsize
    )

    expected_file_size = (
        offset
        + expected_payload_bytes
    )

    actual_file_size = file_path.stat().st_size

    if actual_file_size != expected_file_size:
        raise ValueError(
            f"\nNormalized file size mismatch\n"
            f"File            : {file_path}\n"
            f"Actual bytes    : {actual_file_size:,}\n"
            f"Expected bytes  : {expected_file_size:,}\n"
            f"Expected shape  : {shape}\n"
            f"Expected dtype  : {dtype}\n"
            f"Header size     : {offset}"
        )


# ============================================================
# 4. Load normalized LOW volume as memmap
# ============================================================

def load_normalized_memmap(
    file_path: Path,
    shape: tuple[int, int, int] = VOLUME_SHAPE,
    dtype: np.dtype = INPUT_DTYPE,
    offset: int = HEADER_SIZE,
) -> np.memmap:
    """
    Load normalized float32 LOW volume as np.memmap.

    Shape:
        [H, W, D] = [200, 200, 512]
    """

    validate_normalized_file(
        file_path=file_path,
        shape=shape,
        dtype=dtype,
        offset=offset,
    )

    volume = np.memmap(
        filename=file_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="C",
    )

    return volume


# ============================================================
# 5. Get adjacent slice indices
# ============================================================

def get_adjacent_indices(
    depth_index: int,
    depth_size: int,
) -> tuple[int, int, int]:
    """
    Return:
        previous_depth,
        current_depth,
        next_depth

    Boundary handling:
        replicate padding

    Examples:
        d = 0
        -> (0, 0, 1)

        d = 100
        -> (99, 100, 101)

        d = 511
        -> (510, 511, 511)
    """

    if depth_index < 0 or depth_index >= depth_size:
        raise IndexError(
            f"Invalid depth index: {depth_index}. "
            f"Valid range: 0 ~ {depth_size - 1}"
        )

    previous_depth = max(
        depth_index - 1,
        0,
    )

    current_depth = depth_index

    next_depth = min(
        depth_index + 1,
        depth_size - 1,
    )

    return (
        previous_depth,
        current_depth,
        next_depth,
    )


# ============================================================
# 6. Create one 3-adjacent-slice LOW file
# ============================================================

def create_adjacent_slice_file(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Convert one normalized LOW volume:

        Input:
            [H, W, D]
            [200, 200, 512]

    into:

        Output:
            [D, C, H, W]
            [512, 3, 200, 200]

    Channels:
        0 = previous LOW slice
        1 = current LOW slice
        2 = next LOW slice

    Output file format:
        [original 48-byte header]
        +
        [float32 grouped payload]
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # --------------------------------------------------------
    # Read original header
    # --------------------------------------------------------

    header = read_header(input_path)

    # --------------------------------------------------------
    # Open normalized input LOW volume
    # --------------------------------------------------------

    volume = load_normalized_memmap(
        input_path
    )

    height, width, depth = VOLUME_SHAPE

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Write 48-byte header
    # --------------------------------------------------------

    with output_path.open("wb") as f:
        f.write(header)

    # --------------------------------------------------------
    # Create output payload directly on disk
    #
    # Shape:
    #     [D, 3, H, W]
    # --------------------------------------------------------

    grouped = np.memmap(
        filename=output_path,
        dtype=OUTPUT_DTYPE,
        mode="r+",
        offset=HEADER_SIZE,
        shape=OUTPUT_SHAPE,
        order="C",
    )

    # --------------------------------------------------------
    # Process each depth slice
    # --------------------------------------------------------

    for d in range(depth):

        prev_d, curr_d, next_d = (
            get_adjacent_indices(
                depth_index=d,
                depth_size=depth,
            )
        )

        # Channel 0: previous LOW slice
        grouped[d, 0, :, :] = volume[
            :,
            :,
            prev_d,
        ]

        # Channel 1: current LOW slice
        grouped[d, 1, :, :] = volume[
            :,
            :,
            curr_d,
        ]

        # Channel 2: next LOW slice
        grouped[d, 2, :, :] = volume[
            :,
            :,
            next_d,
        ]

    # Write all buffered data to disk.
    grouped.flush()

    # --------------------------------------------------------
    # Collect statistics
    # --------------------------------------------------------

    statistics = {
        "file_name": input_path.name,
        "input_shape": list(VOLUME_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "dtype": str(np.dtype(OUTPUT_DTYPE)),
        "min": float(grouped.min()),
        "max": float(grouped.max()),
        "mean": float(grouped.mean()),
    }

    del grouped
    del volume

    return statistics


# ============================================================
# 7. Process one LOW directory
# ============================================================

def process_directory(
    input_dir: Path,
    output_dir: Path,
    label: str,
) -> list[dict]:
    """
    Convert every normalized LOW .bin file
    in one directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = get_bin_files(input_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print(
        f"CREATING 3-ADJACENT-SLICE LOW DATA: "
        f"{label}"
    )
    print("=" * 70)

    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Files       : {len(files)}")
    print(f"Input shape : {VOLUME_SHAPE}")
    print(f"Output shape: {OUTPUT_SHAPE}")
    print(f"Dtype       : {np.dtype(OUTPUT_DTYPE)}")

    all_statistics = []

    for index, input_path in enumerate(
        files,
        start=1,
    ):
        output_path = (
            output_dir / input_path.name
        )

        print(
            f"[{index:3d}/{len(files)}] "
            f"{input_path.name}"
        )

        stats = create_adjacent_slice_file(
            input_path=input_path,
            output_path=output_path,
        )

        stats["label"] = label

        all_statistics.append(stats)

    return all_statistics


# ============================================================
# 8. Verify one generated file
# ============================================================

def verify_output_file(
    file_path: Path,
) -> None:
    """
    Verify one generated adjacent-slice LOW file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Output file not found: {file_path}"
        )

    expected_elements = int(
        np.prod(OUTPUT_SHAPE)
    )

    expected_file_size = (
        HEADER_SIZE
        + expected_elements
        * np.dtype(OUTPUT_DTYPE).itemsize
    )

    actual_file_size = (
        file_path.stat().st_size
    )

    if actual_file_size != expected_file_size:
        raise ValueError(
            f"Output verification failed.\n"
            f"File           : {file_path}\n"
            f"Actual bytes   : {actual_file_size:,}\n"
            f"Expected bytes : {expected_file_size:,}"
        )

    grouped = np.memmap(
        filename=file_path,
        dtype=OUTPUT_DTYPE,
        mode="r",
        offset=HEADER_SIZE,
        shape=OUTPUT_SHAPE,
        order="C",
    )

    print("\n" + "=" * 70)
    print("3-ADJACENT-SLICE LOW FILE VERIFICATION")
    print("=" * 70)

    print(f"File  : {file_path}")
    print(f"Shape : {grouped.shape}")
    print(f"Dtype : {grouped.dtype}")

    print("\nGlobal statistics")
    print(f"  Min  : {grouped.min():.6f}")
    print(f"  Max  : {grouped.max():.6f}")
    print(f"  Mean : {grouped.mean():.6f}")

    print("\nBoundary structure")

    print(
        "  depth 0   : "
        "[slice 0, slice 0, slice 1]"
    )

    print(
        "  depth 511 : "
        "[slice 510, slice 511, slice 511]"
    )

    print("\nExample sample shapes")

    print(
        f"  grouped[0].shape   : "
        f"{grouped[0].shape}"
    )

    print(
        f"  grouped[100].shape : "
        f"{grouped[100].shape}"
    )

    print(
        f"  grouped[511].shape : "
        f"{grouped[511].shape}"
    )

    del grouped


# ============================================================
# 9. Save configuration
# ============================================================

def save_config() -> None:
    """
    Save three-adjacent-slice LOW data settings.
    """

    config = {
        "method": "three_adjacent_low_slices",
        "input_directories": {
            "train_low": str(
                NORM_TRAIN_LOW_DIR
            ),
            "test_low": str(
                NORM_TEST_LOW_DIR
            ),
        },
        "output_directories": {
            "train_low": str(
                OUTPUT_TRAIN_LOW_DIR
            ),
            "test_low": str(
                OUTPUT_TEST_LOW_DIR
            ),
        },
        "input_shape": list(
            VOLUME_SHAPE
        ),
        "input_shape_order": "[H, W, D]",
        "output_shape": list(
            OUTPUT_SHAPE
        ),
        "output_shape_order": "[D, C, H, W]",
        "channels": {
            "0": "previous_low_slice",
            "1": "current_low_slice",
            "2": "next_low_slice",
        },
        "boundary_handling": "replicate",
        "header_size_bytes": HEADER_SIZE,
        "input_dtype": str(
            np.dtype(INPUT_DTYPE)
        ),
        "output_dtype": str(
            np.dtype(OUTPUT_DTYPE)
        ),
        "clipping": False,
        "normalization_changed": False,
        "common_scale_preserved": True,
        "high_data_duplicated": False,
        "high_target_source": {
            "train": "./data/norm_Train/HIGH",
            "test": "./data/norm_Test/HIGH",
        },
    }

    config_path = (
        DATA_ROOT
        / "three_adjacent_low_slices_config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\nSaved config: {config_path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("PAM THREE ADJACENT LOW SLICES DATA CREATION")
    print("=" * 70)

    print(
        "\nInput normalized LOW data:"
        f"\n  Train: {NORM_TRAIN_LOW_DIR}"
        f"\n  Test : {NORM_TEST_LOW_DIR}"
    )

    print(
        "\nOutput grouped LOW data:"
        f"\n  Train: {OUTPUT_TRAIN_LOW_DIR}"
        f"\n  Test : {OUTPUT_TEST_LOW_DIR}"
    )

    print(
        "\nAdjacent-slice structure:"
        "\n  Channel 0 = previous LOW slice"
        "\n  Channel 1 = current LOW slice"
        "\n  Channel 2 = next LOW slice"
    )

    # --------------------------------------------------------
    # Train LOW
    # --------------------------------------------------------

    process_directory(
        input_dir=NORM_TRAIN_LOW_DIR,
        output_dir=OUTPUT_TRAIN_LOW_DIR,
        label="TRAIN LOW",
    )

    # --------------------------------------------------------
    # Test LOW
    # --------------------------------------------------------

    process_directory(
        input_dir=NORM_TEST_LOW_DIR,
        output_dir=OUTPUT_TEST_LOW_DIR,
        label="TEST LOW",
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    save_config()

    # --------------------------------------------------------
    # Verify first Train LOW output
    # --------------------------------------------------------

    output_files = get_bin_files(
        OUTPUT_TRAIN_LOW_DIR
    )

    verify_output_file(
        output_files[0]
    )

    print("\n" + "=" * 70)
    print("3-ADJACENT LOW SLICE DATA CREATION COMPLETE")
    print("=" * 70)

    print(
        f"\nTrain output: {OUTPUT_TRAIN_LOW_DIR}"
    )

    print(
        f"Test output : {OUTPUT_TEST_LOW_DIR}"
    )

    print(
        "\nHIGH targets remain unchanged:"
        "\n  ./data/norm_Train/HIGH"
        "\n  ./data/norm_Test/HIGH"
    )

    print(
        "\nImportant:"
        "\n- Only LOW data is converted to 3 adjacent slices."
        "\n- HIGH data is not duplicated."
        "\n- Input normalized data remains unchanged."
        "\n- Common scale 2983.0 is preserved."
        "\n- No clipping is applied."
        "\n- Values greater than 1.0 are preserved."
        "\n- Output dtype is float32."
        "\n- Output shape per file is [512, 3, 200, 200]."
        "\n- Boundary handling uses replicate padding."
    )


if __name__ == "__main__":
    main()