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

# Output 3-adjacent Y-Z LOW volumes.
# Keep these separate from the existing X-Z files to avoid overwriting them.
OUTPUT_TRAIN_LOW_DIR = DATA_ROOT / "3d_yz_Train" / "LOW"
OUTPUT_TEST_LOW_DIR = DATA_ROOT / "3d_yz_Test" / "LOW"

# Original normalized PAM volume
# Shape order: [X, Y, Z]
# Shape      : [200, 200, 512]
VOLUME_SHAPE = (200, 200, 512)

INPUT_DTYPE = np.float32
OUTPUT_DTYPE = np.float32

HEADER_SIZE = 48
NUM_CHANNELS = 3


# ============================================================
# Y-Z plane configuration
# ============================================================

X_SIZE = VOLUME_SHAPE[0]  # 200
Y_SIZE = VOLUME_SHAPE[1]  # 200
Z_SIZE = VOLUME_SHAPE[2]  # 512

# All X positions are preserved.
#
# x = 0   -> [0, 0, 1]
# x = 1   -> [0, 1, 2]
# ...
# x = 199 -> [198, 199, 199]
NUM_YZ_SAMPLES = X_SIZE

# Output shape:
# [N, C, Y, Z]
# [200, 3, 200, 512]
OUTPUT_SHAPE = (
    NUM_YZ_SAMPLES,
    NUM_CHANNELS,
    Y_SIZE,
    Z_SIZE,
)


# ============================================================
# 1. Find .bin files
# ============================================================


def get_bin_files(directory: Path) -> list[Path]:
    """Find and sort all .bin files in a directory."""

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.bin"))

    if not files:
        raise RuntimeError(f"No .bin files found in: {directory}")

    return files


# ============================================================
# 2. Read original 48-byte header
# ============================================================


def read_header(
    file_path: Path,
    header_size: int = HEADER_SIZE,
) -> bytes:
    """Read the original 48-byte header."""

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
    Validate normalized LOW .bin file.

    Expected format:
        [48-byte header] + [float32 payload]

    Payload shape:
        [X, Y, Z] = [200, 200, 512]
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dtype = np.dtype(dtype)
    expected_voxels = int(np.prod(shape))
    expected_payload_bytes = expected_voxels * dtype.itemsize
    expected_file_size = offset + expected_payload_bytes
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
    Load normalized float32 LOW volume.

    Shape:
        [X, Y, Z] = [200, 200, 512]
    """

    validate_normalized_file(
        file_path=file_path,
        shape=shape,
        dtype=dtype,
        offset=offset,
    )

    return np.memmap(
        filename=file_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="C",
    )


# ============================================================
# 5. Get adjacent X indices
# ============================================================


def get_adjacent_x_indices(
    x_index: int,
    x_size: int,
) -> tuple[int, int, int]:
    """
    Return three adjacent X indices:
        previous_x, current_x, next_x

    Boundary handling:
        replicate padding

    Examples:
        x = 0   -> (0, 0, 1)
        x = 100 -> (99, 100, 101)
        x = 199 -> (198, 199, 199)
    """

    if x_index < 0 or x_index >= x_size:
        raise IndexError(
            f"Invalid X index: {x_index}. "
            f"Valid range: 0 ~ {x_size - 1}"
        )

    previous_x = max(x_index - 1, 0)
    current_x = x_index
    next_x = min(x_index + 1, x_size - 1)

    return previous_x, current_x, next_x


# ============================================================
# 6. Create one 3-adjacent Y-Z LOW file
# ============================================================


def create_adjacent_yz_slice_file(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Convert one normalized LOW PAM volume.

    Input:
        [X, Y, Z] = [200, 200, 512]

    Y-Z plane:
        volume[x, :, :]

    Three adjacent Y-Z planes:
        Channel 0 = volume[x-1, :, :]
        Channel 1 = volume[x,   :, :]
        Channel 2 = volume[x+1, :, :]

    Boundary handling:
        x = 0   -> [0, 0, 1]
        x = 199 -> [198, 199, 199]

    Output:
        [N, C, Y, Z] = [200, 3, 200, 512]

    Output file format:
        [original 48-byte header] + [float32 grouped payload]
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    header = read_header(input_path)
    volume = load_normalized_memmap(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_elements = int(np.prod(OUTPUT_SHAPE))
    expected_file_size = (
        HEADER_SIZE
        + expected_elements * np.dtype(OUTPUT_DTYPE).itemsize
    )

    # Write the original header and preallocate the exact output size.
    # This avoids mapping a file that is still only 48 bytes long.
    with output_path.open("wb") as f:
        f.write(header)
        f.truncate(expected_file_size)

    grouped = np.memmap(
        filename=output_path,
        dtype=OUTPUT_DTYPE,
        mode="r+",
        offset=HEADER_SIZE,
        shape=OUTPUT_SHAPE,
        order="C",
    )

    for x in range(X_SIZE):
        previous_x, current_x, next_x = get_adjacent_x_indices(
            x_index=x,
            x_size=X_SIZE,
        )

        # Each source Y-Z plane is contiguous in C-order [X, Y, Z].
        grouped[x, 0, :, :] = volume[previous_x, :, :]
        grouped[x, 1, :, :] = volume[current_x, :, :]
        grouped[x, 2, :, :] = volume[next_x, :, :]

    grouped.flush()

    statistics = {
        "file_name": input_path.name,
        "plane": "YZ",
        "adjacent_axis": "X",
        "input_shape": list(VOLUME_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
        "number_of_samples": NUM_YZ_SAMPLES,
        "x_index_range": [0, X_SIZE - 1],
        "boundary_handling": "replicate",
        "dtype": str(np.dtype(OUTPUT_DTYPE)),
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
    """Convert every normalized LOW .bin file into 3-adjacent Y-Z data."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = get_bin_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"CREATING 3-ADJACENT Y-Z LOW DATA: {label}")
    print("=" * 70)
    print(f"Input        : {input_dir}")
    print(f"Output       : {output_dir}")
    print(f"Files        : {len(files)}")
    print(f"Input shape  : {VOLUME_SHAPE}")
    print("Input order  : [X, Y, Z]")
    print("Plane        : Y-Z")
    print("Adjacent axis: X")
    print(f"Output shape : {OUTPUT_SHAPE}")
    print("Output order : [N, C, Y, Z]")
    print("X range      : 0 ~ 199")
    print("Boundary     : replicate padding")
    print(f"Dtype        : {np.dtype(OUTPUT_DTYPE)}")

    all_statistics: list[dict] = []

    for index, input_path in enumerate(files, start=1):
        output_path = output_dir / input_path.name

        print(f"[{index:3d}/{len(files)}] {input_path.name}")

        stats = create_adjacent_yz_slice_file(
            input_path=input_path,
            output_path=output_path,
        )
        stats["label"] = label
        all_statistics.append(stats)

    return all_statistics


# ============================================================
# 8. Verify one generated file
# ============================================================


def verify_output_file(file_path: Path) -> None:
    """Verify one generated 3-adjacent Y-Z LOW file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Output file not found: {file_path}")

    expected_elements = int(np.prod(OUTPUT_SHAPE))
    expected_file_size = (
        HEADER_SIZE
        + expected_elements * np.dtype(OUTPUT_DTYPE).itemsize
    )
    actual_file_size = file_path.stat().st_size

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

    boundary_start_error = float(
        np.max(np.abs(grouped[0, 0] - grouped[0, 1]))
    )
    boundary_end_error = float(
        np.max(np.abs(grouped[-1, 1] - grouped[-1, 2]))
    )

    print("\n" + "=" * 70)
    print("3-ADJACENT Y-Z LOW FILE VERIFICATION")
    print("=" * 70)
    print(f"File  : {file_path}")
    print(f"Shape : {grouped.shape}")
    print(f"Dtype : {grouped.dtype}")

    print("\nGlobal statistics")
    print(f"  Min  : {grouped.min():.6f}")
    print(f"  Max  : {grouped.max():.6f}")
    print(f"  Mean : {grouped.mean():.6f}")

    print("\nY-Z adjacent structure")
    print("  X = 0   : [X=0, X=0, X=1]")
    print("  X = 1   : [X=0, X=1, X=2]")
    print("  X = 100 : [X=99, X=100, X=101]")
    print("  X = 198 : [X=197, X=198, X=199]")
    print("  X = 199 : [X=198, X=199, X=199]")

    print("\nBoundary verification")
    print(f"  X=0   Channel 0 == Channel 1 max error: {boundary_start_error:.6e}")
    print(f"  X=199 Channel 1 == Channel 2 max error: {boundary_end_error:.6e}")

    print("\nExample sample shapes")
    print(f"  grouped[0].shape   : {grouped[0].shape}")
    print(f"  grouped[100].shape : {grouped[100].shape}")
    print(f"  grouped[199].shape : {grouped[199].shape}")

    del grouped


# ============================================================
# 9. Save configuration
# ============================================================


def save_config() -> None:
    """Save 3-adjacent Y-Z LOW data settings."""

    config = {
        "method": "three_adjacent_yz_low_slices",
        "plane": "YZ",
        "adjacent_axis": "X",
        "input_directories": {
            "train_low": str(NORM_TRAIN_LOW_DIR),
            "test_low": str(NORM_TEST_LOW_DIR),
        },
        "output_directories": {
            "train_low": str(OUTPUT_TRAIN_LOW_DIR),
            "test_low": str(OUTPUT_TEST_LOW_DIR),
        },
        "input_shape": list(VOLUME_SHAPE),
        "input_shape_order": "[X, Y, Z]",
        "yz_slice_shape": [Y_SIZE, Z_SIZE],
        "output_shape": list(OUTPUT_SHAPE),
        "output_shape_order": "[N, C, Y, Z]",
        "number_of_samples_per_volume": NUM_YZ_SAMPLES,
        "x_index_range": [0, X_SIZE - 1],
        "channels": {
            "0": "previous_x_yz_low_slice",
            "1": "current_x_yz_low_slice",
            "2": "next_x_yz_low_slice",
        },
        "boundary_handling": "replicate_padding",
        "boundary_examples": {
            "x_0": [0, 0, 1],
            "x_199": [198, 199, 199],
        },
        "header_size_bytes": HEADER_SIZE,
        "input_dtype": str(np.dtype(INPUT_DTYPE)),
        "output_dtype": str(np.dtype(OUTPUT_DTYPE)),
        "clipping": False,
        "normalization_changed": False,
        "common_scale_preserved": True,
        "high_data_duplicated": False,
        "high_target_definition": "HIGH[x, :, :]",
        "high_target_source": {
            "train": "./data/norm_Train/HIGH",
            "test": "./data/norm_Test/HIGH",
        },
        "map_visualization": {
            "predicted_volume_shape": "[X, Y, Z] = [200, 200, 512]",
            "output_plane": "XY",
            "method": (
                "Stack all 200 predicted Y-Z planes in X order to reconstruct "
                "the [X, Y, Z] volume, then perform maximum projection over Z."
            ),
            "numpy_operation": "np.max(predicted_volume, axis=2)",
        },
    }

    config_path = DATA_ROOT / "three_adjacent_yz_low_slices_config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\nSaved config: {config_path}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("\n" + "=" * 70)
    print("PAM THREE ADJACENT Y-Z LOW SLICES DATA CREATION")
    print("=" * 70)

    print(
        "\nOriginal volume:"
        "\n  Shape order = [X, Y, Z]"
        "\n  Shape       = [200, 200, 512]"
    )

    print(
        "\nTraining plane:"
        "\n  Y-Z plane"
        "\n  Fixed X index"
    )

    print(
        "\nAdjacent structure:"
        "\n  Channel 0 = Y-Z plane at X - 1"
        "\n  Channel 1 = Y-Z plane at X"
        "\n  Channel 2 = Y-Z plane at X + 1"
    )

    print(
        "\nBoundary handling:"
        "\n  Replicate padding"
        "\n  X=0   -> [0, 0, 1]"
        "\n  X=199 -> [198, 199, 199]"
    )

    print(
        "\nAll Y-Z planes are preserved:"
        "\n  X range = 0 ~ 199"
        "\n  Total   = 200 samples per volume"
    )

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

    process_directory(
        input_dir=NORM_TRAIN_LOW_DIR,
        output_dir=OUTPUT_TRAIN_LOW_DIR,
        label="TRAIN LOW",
    )

    process_directory(
        input_dir=NORM_TEST_LOW_DIR,
        output_dir=OUTPUT_TEST_LOW_DIR,
        label="TEST LOW",
    )

    save_config()

    output_files = get_bin_files(OUTPUT_TRAIN_LOW_DIR)
    verify_output_file(output_files[0])

    print("\n" + "=" * 70)
    print("3-ADJACENT Y-Z LOW SLICE DATA CREATION COMPLETE")
    print("=" * 70)

    print(f"\nTrain output: {OUTPUT_TRAIN_LOW_DIR}")
    print(f"Test output : {OUTPUT_TEST_LOW_DIR}")

    print(
        "\nHIGH targets remain unchanged:"
        "\n  ./data/norm_Train/HIGH"
        "\n  ./data/norm_Test/HIGH"
    )

    print(
        "\nImportant:"
        "\n- Training plane is Y-Z, not X-Z."
        "\n- Adjacent context is taken along the X axis."
        "\n- All 200 Y-Z planes are preserved."
        "\n- X=0 uses [0, 0, 1]."
        "\n- X=199 uses [198, 199, 199]."
        "\n- Each volume produces 200 samples."
        "\n- Each input sample shape is [3, 200, 512]."
        "\n- HIGH target is HIGH[x, :, :]."
        "\n- Output shape per file is [200, 3, 200, 512]."
        "\n- Final predicted volume is reconstructed directly as [X, Y, Z]."
        "\n- X-Y MAP is np.max(predicted_volume, axis=2)."
        "\n- Common scale 2983.0 is preserved."
        "\n- No clipping is applied."
        "\n- Values greater than 1.0 are preserved."
        "\n- Output dtype is float32."
    )


if __name__ == "__main__":
    main()
