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

# Output 3-adjacent X-Z LOW volumes
OUTPUT_TRAIN_LOW_DIR = DATA_ROOT / "3d_Train" / "LOW"
OUTPUT_TEST_LOW_DIR = DATA_ROOT / "3d_Test" / "LOW"

# Original normalized PAM volume
#
# Shape order:
#     [X, Y, Z]
#
# Shape:
#     [200, 200, 512]
#
VOLUME_SHAPE = (200, 200, 512)

INPUT_DTYPE = np.float32
OUTPUT_DTYPE = np.float32

HEADER_SIZE = 48

NUM_CHANNELS = 3


# ============================================================
# X-Z plane configuration
# ============================================================

X_SIZE = VOLUME_SHAPE[0]   # 200
Y_SIZE = VOLUME_SHAPE[1]   # 200
Z_SIZE = VOLUME_SHAPE[2]   # 512

# All Y planes are preserved.
#
# y = 0:
#     [0, 0, 1]
#
# y = 1:
#     [0, 1, 2]
#
# ...
#
# y = 199:
#     [198, 199, 199]
#
NUM_XZ_SAMPLES = Y_SIZE

# Output shape:
#
# [N, C, X, Z]
# [200, 3, 200, 512]
#
OUTPUT_SHAPE = (
    NUM_XZ_SAMPLES,
    NUM_CHANNELS,
    X_SIZE,
    Z_SIZE,
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
    Validate normalized LOW .bin file.

    Expected format:

        [48-byte header]
        +
        [float32 payload]

    Payload shape:

        [X, Y, Z]
        [200, 200, 512]
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    dtype = np.dtype(dtype)

    expected_voxels = int(
        np.prod(shape)
    )

    expected_payload_bytes = (
        expected_voxels
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

        [X, Y, Z]
        [200, 200, 512]
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
# 5. Get adjacent Y indices
# ============================================================

def get_adjacent_y_indices(
    y_index: int,
    y_size: int,
) -> tuple[int, int, int]:
    """
    Return three adjacent Y indices:

        previous_y
        current_y
        next_y

    Boundary handling:
        replicate padding

    Examples:

        y = 0
        -> (0, 0, 1)

        y = 100
        -> (99, 100, 101)

        y = 199
        -> (198, 199, 199)
    """

    if y_index < 0 or y_index >= y_size:
        raise IndexError(
            f"Invalid Y index: {y_index}. "
            f"Valid range: 0 ~ {y_size - 1}"
        )

    previous_y = max(
        y_index - 1,
        0,
    )

    current_y = y_index

    next_y = min(
        y_index + 1,
        y_size - 1,
    )

    return (
        previous_y,
        current_y,
        next_y,
    )


# ============================================================
# 6. Create one 3-adjacent X-Z LOW file
# ============================================================

def create_adjacent_xz_slice_file(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Convert one normalized LOW PAM volume.

    Input:
        [X, Y, Z]
        [200, 200, 512]

    X-Z plane:
        volume[:, y, :]

    Three adjacent X-Z planes:

        Channel 0:
            volume[:, y-1, :]

        Channel 1:
            volume[:, y, :]

        Channel 2:
            volume[:, y+1, :]

    Boundary handling:

        y = 0:
            [0, 0, 1]

        y = 199:
            [198, 199, 199]

    Output:

        [N, C, X, Z]
        [200, 3, 200, 512]

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

    header = read_header(
        input_path
    )

    # --------------------------------------------------------
    # Load normalized LOW volume
    #
    # [X, Y, Z]
    # [200, 200, 512]
    # --------------------------------------------------------

    volume = load_normalized_memmap(
        input_path
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Write original 48-byte header
    # --------------------------------------------------------

    with output_path.open("wb") as f:
        f.write(header)

    # --------------------------------------------------------
    # Create output memmap
    #
    # Shape:
    #
    # [N, C, X, Z]
    # [200, 3, 200, 512]
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
    # Process all Y positions
    #
    # y = 0 ~ 199
    # --------------------------------------------------------

    for y in range(Y_SIZE):

        previous_y, current_y, next_y = (
            get_adjacent_y_indices(
                y_index=y,
                y_size=Y_SIZE,
            )
        )

        # ----------------------------------------------------
        # Channel 0:
        # Previous X-Z plane
        #
        # shape:
        # [200, 512]
        # ----------------------------------------------------

        grouped[
            y,
            0,
            :,
            :,
        ] = volume[
            :,
            previous_y,
            :,
        ]

        # ----------------------------------------------------
        # Channel 1:
        # Current X-Z plane
        # ----------------------------------------------------

        grouped[
            y,
            1,
            :,
            :,
        ] = volume[
            :,
            current_y,
            :,
        ]

        # ----------------------------------------------------
        # Channel 2:
        # Next X-Z plane
        # ----------------------------------------------------

        grouped[
            y,
            2,
            :,
            :,
        ] = volume[
            :,
            next_y,
            :,
        ]

    # Write buffered data to disk.
    grouped.flush()

    # --------------------------------------------------------
    # Collect statistics
    # --------------------------------------------------------

    statistics = {
        "file_name": input_path.name,

        "plane": "XZ",

        "adjacent_axis": "Y",

        "input_shape": list(
            VOLUME_SHAPE
        ),

        "output_shape": list(
            OUTPUT_SHAPE
        ),

        "number_of_samples": (
            NUM_XZ_SAMPLES
        ),

        "y_index_range": [
            0,
            Y_SIZE - 1,
        ],

        "boundary_handling": (
            "replicate"
        ),

        "dtype": str(
            np.dtype(OUTPUT_DTYPE)
        ),

        "min": float(
            grouped.min()
        ),

        "max": float(
            grouped.max()
        ),

        "mean": float(
            grouped.mean()
        ),
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
    into 3-adjacent X-Z slice data.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = get_bin_files(
        input_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)

    print(
        f"CREATING 3-ADJACENT X-Z LOW DATA: "
        f"{label}"
    )

    print("=" * 70)

    print(
        f"Input        : {input_dir}"
    )

    print(
        f"Output       : {output_dir}"
    )

    print(
        f"Files        : {len(files)}"
    )

    print(
        f"Input shape  : {VOLUME_SHAPE}"
    )

    print(
        "Input order  : [X, Y, Z]"
    )

    print(
        "Plane        : X-Z"
    )

    print(
        "Adjacent axis: Y"
    )

    print(
        f"Output shape : {OUTPUT_SHAPE}"
    )

    print(
        "Output order : [N, C, X, Z]"
    )

    print(
        "Y range      : 0 ~ 199"
    )

    print(
        "Boundary     : replicate padding"
    )

    print(
        f"Dtype        : "
        f"{np.dtype(OUTPUT_DTYPE)}"
    )

    all_statistics = []

    for index, input_path in enumerate(
        files,
        start=1,
    ):

        output_path = (
            output_dir
            / input_path.name
        )

        print(
            f"[{index:3d}/{len(files)}] "
            f"{input_path.name}"
        )

        stats = (
            create_adjacent_xz_slice_file(
                input_path=input_path,
                output_path=output_path,
            )
        )

        stats["label"] = label

        all_statistics.append(
            stats
        )

    return all_statistics


# ============================================================
# 8. Verify one generated file
# ============================================================

def verify_output_file(
    file_path: Path,
) -> None:
    """
    Verify one generated 3-adjacent X-Z LOW file.
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
        * np.dtype(
            OUTPUT_DTYPE
        ).itemsize
    )

    actual_file_size = (
        file_path.stat().st_size
    )

    if (
        actual_file_size
        != expected_file_size
    ):
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

    print(
        "3-ADJACENT X-Z LOW FILE VERIFICATION"
    )

    print("=" * 70)

    print(
        f"File  : {file_path}"
    )

    print(
        f"Shape : {grouped.shape}"
    )

    print(
        f"Dtype : {grouped.dtype}"
    )

    print(
        "\nGlobal statistics"
    )

    print(
        f"  Min  : "
        f"{grouped.min():.6f}"
    )

    print(
        f"  Max  : "
        f"{grouped.max():.6f}"
    )

    print(
        f"  Mean : "
        f"{grouped.mean():.6f}"
    )

    print(
        "\nX-Z adjacent structure"
    )

    print(
        "  Y = 0   : "
        "[Y=0, Y=0, Y=1]"
    )

    print(
        "  Y = 1   : "
        "[Y=0, Y=1, Y=2]"
    )

    print(
        "  Y = 100 : "
        "[Y=99, Y=100, Y=101]"
    )

    print(
        "  Y = 198 : "
        "[Y=197, Y=198, Y=199]"
    )

    print(
        "  Y = 199 : "
        "[Y=198, Y=199, Y=199]"
    )

    print(
        "\nExample sample shapes"
    )

    print(
        f"  grouped[0].shape   : "
        f"{grouped[0].shape}"
    )

    print(
        f"  grouped[100].shape : "
        f"{grouped[100].shape}"
    )

    print(
        f"  grouped[199].shape : "
        f"{grouped[199].shape}"
    )

    del grouped


# ============================================================
# 9. Save configuration
# ============================================================

def save_config() -> None:
    """
    Save 3-adjacent X-Z LOW data settings.
    """

    config = {
        "method": (
            "three_adjacent_xz_low_slices"
        ),

        "plane": "XZ",

        "adjacent_axis": "Y",

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

        "input_shape_order": (
            "[X, Y, Z]"
        ),

        "xz_slice_shape": [
            X_SIZE,
            Z_SIZE,
        ],

        "output_shape": list(
            OUTPUT_SHAPE
        ),

        "output_shape_order": (
            "[N, C, X, Z]"
        ),

        "number_of_samples_per_volume": (
            NUM_XZ_SAMPLES
        ),

        "y_index_range": [
            0,
            Y_SIZE - 1,
        ],

        "channels": {
            "0": (
                "previous_y_xz_low_slice"
            ),

            "1": (
                "current_y_xz_low_slice"
            ),

            "2": (
                "next_y_xz_low_slice"
            ),
        },

        "boundary_handling": (
            "replicate_padding"
        ),

        "boundary_examples": {
            "y_0": [
                0,
                0,
                1,
            ],

            "y_199": [
                198,
                199,
                199,
            ],
        },

        "header_size_bytes": (
            HEADER_SIZE
        ),

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

        "high_target_definition": (
            "HIGH[:, y, :]"
        ),

        "high_target_source": {
            "train": (
                "./data/norm_Train/HIGH"
            ),

            "test": (
                "./data/norm_Test/HIGH"
            ),
        },

        "map_visualization": {
            "predicted_volume_shape": (
                "[X, Y, Z] = "
                "[200, 200, 512]"
            ),

            "output_plane": "XY",

            "method": (
                "Reconstruct predicted "
                "X-Y-Z volume from all "
                "200 predicted X-Z planes, "
                "then perform maximum "
                "projection over Z axis."
            ),

            "numpy_operation": (
                "np.max("
                "predicted_volume, "
                "axis=2"
                ")"
            ),
        },
    }

    config_path = (
        DATA_ROOT
        / "three_adjacent_xz_low_slices_config.json"
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
        f"\nSaved config: "
        f"{config_path}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n" + "=" * 70)

    print(
        "PAM THREE ADJACENT X-Z "
        "LOW SLICES DATA CREATION"
    )

    print("=" * 70)

    print(
        "\nOriginal volume:"
        "\n  Shape order = [X, Y, Z]"
        "\n  Shape       = [200, 200, 512]"
    )

    print(
        "\nTraining plane:"
        "\n  X-Z plane"
        "\n  Fixed Y index"
    )

    print(
        "\nAdjacent structure:"
        "\n  Channel 0 = X-Z plane at Y - 1"
        "\n  Channel 1 = X-Z plane at Y"
        "\n  Channel 2 = X-Z plane at Y + 1"
    )

    print(
        "\nBoundary handling:"
        "\n  Replicate padding"
        "\n  Y=0   -> [0, 0, 1]"
        "\n  Y=199 -> [198, 199, 199]"
    )

    print(
        "\nAll X-Z planes are preserved:"
        "\n  Y range = 0 ~ 199"
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
    # Verify first Train output
    # --------------------------------------------------------

    output_files = get_bin_files(
        OUTPUT_TRAIN_LOW_DIR
    )

    verify_output_file(
        output_files[0]
    )

    print("\n" + "=" * 70)

    print(
        "3-ADJACENT X-Z LOW SLICE "
        "DATA CREATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nTrain output: "
        f"{OUTPUT_TRAIN_LOW_DIR}"
    )

    print(
        f"Test output : "
        f"{OUTPUT_TEST_LOW_DIR}"
    )

    print(
        "\nHIGH targets remain unchanged:"
        "\n  ./data/norm_Train/HIGH"
        "\n  ./data/norm_Test/HIGH"
    )

    print(
        "\nImportant:"
        "\n- Training plane is X-Z, not X-Y."
        "\n- Adjacent context is taken along the Y axis."
        "\n- All 200 X-Z planes are preserved."
        "\n- Y=0 uses [0, 0, 1]."
        "\n- Y=199 uses [198, 199, 199]."
        "\n- Each volume produces 200 samples."
        "\n- Each input sample shape is [3, 200, 512]."
        "\n- HIGH target is HIGH[:, y, :]."
        "\n- Output shape per file is [200, 3, 200, 512]."
        "\n- Final predicted volume can be reconstructed "
        "as [200, 200, 512]."
        "\n- X-Y MAP can then be calculated with "
        "np.max(predicted_volume, axis=2)."
        "\n- Common scale 2983.0 is preserved."
        "\n- No clipping is applied."
        "\n- Values greater than 1.0 are preserved."
        "\n- Output dtype is float32."
    )


if __name__ == "__main__":
    main()