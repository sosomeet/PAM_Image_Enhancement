from pathlib import Path
import json

import numpy as np


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("./data")

TRAIN_DIR = DATA_ROOT / "Train"
TEST_DIR = DATA_ROOT / "Test"

NORM_TRAIN_DIR = DATA_ROOT / "norm_Train"
NORM_TEST_DIR = DATA_ROOT / "norm_Test"

# Original PAM volume configuration
VOLUME_SHAPE = (200, 200, 512)  # [H, W, D]
INPUT_DTYPE = np.uint16
OUTPUT_DTYPE = np.float32

HEADER_SIZE = 48

# Common scale calculation
ROBUST_PERCENTILE = 99.9

# uint16 possible values: 0 ~ 65535
UINT16_MAX = np.iinfo(np.uint16).max
HISTOGRAM_SIZE = UINT16_MAX + 1

EPS = 1e-8


# ============================================================
# 1. Load original .bin volume
# ============================================================

def load_bin_volume(
    file_path: Path,
    shape=VOLUME_SHAPE,
    dtype=INPUT_DTYPE,
    offset=HEADER_SIZE,
) -> np.ndarray:
    """
    Load original PAM .bin file.

    File format:
        [48-byte header] + [uint16 volume]

    Returns:
        volume with shape [H, W, D]
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
            f"\nData size mismatch\n"
            f"File          : {file_path}\n"
            f"Raw size      : {raw.size:,}\n"
            f"Expected size : {expected_size:,}\n"
            f"Expected shape: {shape}"
        )

    return raw.reshape(shape)


# ============================================================
# 2. Read original header
# ============================================================

def read_header(
    file_path: Path,
    header_size: int = HEADER_SIZE,
) -> bytes:
    """
    Read the original 48-byte header.
    """

    with open(file_path, "rb") as f:
        header = f.read(header_size)

    if len(header) != header_size:
        raise ValueError(
            f"Header size mismatch: {file_path}\n"
            f"Expected: {header_size} bytes\n"
            f"Actual  : {len(header)} bytes"
        )

    return header


# ============================================================
# 3. Find .bin files
# ============================================================

def get_bin_files(directory: Path):
    """
    Find and sort all .bin files.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = sorted(directory.glob("*.bin"))

    if len(files) == 0:
        raise RuntimeError(
            f"No .bin files found in: {directory}"
        )

    return files


# ============================================================
# 4. Compute exact global P99.9 from Train HIGH
# ============================================================

def compute_train_high_global_scale(
    percentile: float = ROBUST_PERCENTILE,
) -> float:
    """
    Compute exact global percentile over all voxels
    from Train/HIGH only.

    The resulting value is used as ONE common scale for:

        Train/HIGH
        Train/LOW
        Test/HIGH
        Test/LOW

    Because the original dtype is uint16, an exact histogram
    with 65,536 bins can be accumulated efficiently.
    """

    high_dir = TRAIN_DIR / "HIGH"

    high_files = get_bin_files(high_dir)

    histogram = np.zeros(
        HISTOGRAM_SIZE,
        dtype=np.int64,
    )

    total_voxels = 0

    print("\n" + "=" * 70)
    print("CALCULATING COMMON GLOBAL SCALE")
    print("=" * 70)

    print(f"Source     : {high_dir}")
    print(f"Files      : {len(high_files)}")
    print(f"Percentile : P{percentile}")

    for index, file_path in enumerate(
        high_files,
        start=1,
    ):
        print(
            f"[{index:3d}/{len(high_files)}] "
            f"{file_path.name}"
        )

        volume = load_bin_volume(file_path)

        current_histogram = np.bincount(
            volume.reshape(-1),
            minlength=HISTOGRAM_SIZE,
        )

        histogram += current_histogram
        total_voxels += volume.size

        del volume
        del current_histogram

    # Cumulative histogram
    cumulative = np.cumsum(histogram)

    # Exact percentile target rank
    target_rank = (
        percentile / 100.0
    ) * (total_voxels - 1)

    scale = np.searchsorted(
        cumulative,
        target_rank + 1,
        side="left",
    )

    scale = float(scale)

    if scale <= EPS:
        raise ValueError(
            f"Invalid common scale: {scale}"
        )

    print("\n" + "-" * 70)
    print(f"COMMON SCALE = {scale:.6f}")
    print("-" * 70)

    return scale


# ============================================================
# 5. Common global fixed scaling without clipping
# ============================================================

def normalize_common_global(
    volume: np.ndarray,
    common_scale: float,
) -> np.ndarray:
    """
    Common global fixed scaling WITHOUT clipping.

    Formula:
        normalized = raw / common_scale

    Important:
        Values greater than 1.0 are preserved.

    Example:
        raw = 15000
        scale = 3000
        normalized = 5.0
    """

    if common_scale <= EPS:
        raise ValueError(
            f"Invalid common scale: {common_scale}"
        )

    volume = volume.astype(
        np.float32,
        copy=False,
    )

    normalized = volume / common_scale

    # IMPORTANT:
    # Do NOT use np.clip().
    #
    # Values > 1.0 contain valid intensity information
    # and must be preserved.

    return normalized.astype(
        np.float32,
        copy=False,
    )


# ============================================================
# 6. Denormalization
# ============================================================

def denormalize_common_global(
    normalized_volume: np.ndarray,
    common_scale: float,
) -> np.ndarray:
    """
    Restore normalized volume to the original raw intensity domain.

    Formula:
        raw = normalized * common_scale
    """

    return (
        normalized_volume.astype(np.float32)
        * common_scale
    )


# ============================================================
# 7. Save normalized .bin
# ============================================================

def save_normalized_bin(
    output_path: Path,
    normalized_volume: np.ndarray,
    header: bytes,
):
    """
    Save:

        [original 48-byte header]
        +
        [float32 normalized volume]

    Note:
        Original payload : uint16
        Normalized payload: float32
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    volume = normalized_volume.astype(
        OUTPUT_DTYPE,
        copy=False,
    )

    with open(output_path, "wb") as f:
        f.write(header)
        volume.tofile(f)


# ============================================================
# 8. Normalize one directory
# ============================================================

def normalize_directory(
    input_dir: Path,
    output_dir: Path,
    common_scale: float,
    label: str,
):
    """
    Normalize all .bin files in one directory
    using exactly the same common scale.
    """

    files = get_bin_files(input_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print(f"NORMALIZING {label}")
    print("=" * 70)

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Scale : {common_scale:.6f}")

    statistics = []

    for index, input_path in enumerate(
        files,
        start=1,
    ):
        output_path = output_dir / input_path.name

        print(
            f"[{index:3d}/{len(files)}] "
            f"{input_path.name}"
        )

        # --------------------------------------------
        # Load raw uint16 volume
        # --------------------------------------------
        volume = load_bin_volume(input_path)

        # --------------------------------------------
        # Preserve original 48-byte header
        # --------------------------------------------
        header = read_header(input_path)

        # --------------------------------------------
        # Raw statistics
        # --------------------------------------------
        raw_min = float(volume.min())
        raw_max = float(volume.max())
        raw_mean = float(volume.mean())

        # --------------------------------------------
        # Common global scaling
        # NO CLIPPING
        # --------------------------------------------
        normalized = normalize_common_global(
            volume=volume,
            common_scale=common_scale,
        )

        # --------------------------------------------
        # Normalized statistics
        # --------------------------------------------
        norm_min = float(normalized.min())
        norm_max = float(normalized.max())
        norm_mean = float(normalized.mean())

        ratio_above_one = float(
            np.mean(normalized > 1.0)
        )

        # --------------------------------------------
        # Save float32 binary
        # --------------------------------------------
        save_normalized_bin(
            output_path=output_path,
            normalized_volume=normalized,
            header=header,
        )

        statistics.append({
            "file_name": input_path.name,
            "label": label,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "norm_mean": norm_mean,
            "ratio_above_one": ratio_above_one,
        })

        del volume
        del normalized

    return statistics


# ============================================================
# 9. Verify normalized output
# ============================================================

def verify_normalized_file(
    file_path: Path,
    common_scale: float,
):
    """
    Verify one normalized float32 .bin file.
    """

    raw = np.fromfile(
        file_path,
        dtype=np.float32,
        offset=HEADER_SIZE,
    )

    expected_size = int(np.prod(VOLUME_SHAPE))

    if raw.size != expected_size:
        raise ValueError(
            f"Verification failed: {file_path}\n"
            f"Actual size  : {raw.size:,}\n"
            f"Expected size: {expected_size:,}"
        )

    volume = raw.reshape(VOLUME_SHAPE)

    reconstructed = denormalize_common_global(
        volume,
        common_scale,
    )

    print("\n" + "=" * 70)
    print("NORMALIZED FILE VERIFICATION")
    print("=" * 70)

    print(f"File     : {file_path}")
    print(f"Shape    : {volume.shape}")
    print(f"Dtype    : {volume.dtype}")

    print("\nNormalized domain")
    print(f"  Min    : {volume.min():.6f}")
    print(f"  Max    : {volume.max():.6f}")
    print(f"  Mean   : {volume.mean():.6f}")

    print("\nReconstructed raw domain")
    print(f"  Min    : {reconstructed.min():.6f}")
    print(f"  Max    : {reconstructed.max():.6f}")
    print(f"  Mean   : {reconstructed.mean():.6f}")

    print(
        f"\nRatio > 1.0: "
        f"{np.mean(volume > 1.0) * 100:.6f}%"
    )


# ============================================================
# 10. Save configuration
# ============================================================

def save_config(
    common_scale: float,
):
    """
    Save normalization settings for training,
    inference and evaluation.
    """

    config = {
        "method": (
            "common_global_fixed_scaling_without_clipping"
        ),
        "common_scale": common_scale,
        "scale_source": "Train/HIGH only",
        "scale_statistic": (
            f"global_P{ROBUST_PERCENTILE}"
        ),
        "clipping": False,
        "formula": "x_norm = x_raw / common_scale",
        "inverse_formula": (
            "x_raw = x_norm * common_scale"
        ),
        "input_dtype": str(
            np.dtype(INPUT_DTYPE)
        ),
        "output_dtype": str(
            np.dtype(OUTPUT_DTYPE)
        ),
        "volume_shape": list(VOLUME_SHAPE),
        "header_size": HEADER_SIZE,
    }

    config_path = (
        DATA_ROOT
        / "common_global_scaling_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nSaved config: {config_path}")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print(
        "PAM COMMON GLOBAL FIXED SCALING "
        "WITHOUT CLIPPING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Compute ONE common scale from Train/HIGH only
    # --------------------------------------------------------

    common_scale = compute_train_high_global_scale(
        percentile=ROBUST_PERCENTILE,
    )

    # --------------------------------------------------------
    # 2. Apply exactly the same scale everywhere
    # --------------------------------------------------------

    normalize_directory(
        input_dir=TRAIN_DIR / "LOW",
        output_dir=NORM_TRAIN_DIR / "LOW",
        common_scale=common_scale,
        label="TRAIN LOW",
    )

    normalize_directory(
        input_dir=TRAIN_DIR / "HIGH",
        output_dir=NORM_TRAIN_DIR / "HIGH",
        common_scale=common_scale,
        label="TRAIN HIGH",
    )

    normalize_directory(
        input_dir=TEST_DIR / "LOW",
        output_dir=NORM_TEST_DIR / "LOW",
        common_scale=common_scale,
        label="TEST LOW",
    )

    normalize_directory(
        input_dir=TEST_DIR / "HIGH",
        output_dir=NORM_TEST_DIR / "HIGH",
        common_scale=common_scale,
        label="TEST HIGH",
    )

    # --------------------------------------------------------
    # 3. Save config
    # --------------------------------------------------------

    save_config(common_scale)

    # --------------------------------------------------------
    # 4. Verify first normalized Train HIGH file
    # --------------------------------------------------------

    normalized_high_files = get_bin_files(
        NORM_TRAIN_DIR / "HIGH"
    )

    verify_normalized_file(
        file_path=normalized_high_files[0],
        common_scale=common_scale,
    )

    print("\n" + "=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)

    print(f"COMMON_SCALE: {common_scale:.6f}")

    print(f"\nTrain output: {NORM_TRAIN_DIR}")
    print(f"Test output : {NORM_TEST_DIR}")

    print(
        "\nImportant:"
        "\n- Same scale for LOW and HIGH."
        "\n- Same scale for Train and Test."
        "\n- No clipping."
        "\n- Values greater than 1.0 are preserved."
        "\n- Output payload dtype is float32."
    )


if __name__ == "__main__":
    main()