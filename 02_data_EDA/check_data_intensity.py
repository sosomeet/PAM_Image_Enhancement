from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("./data")
OUTPUT_DIR = Path("./02_data_EDA/analysis_results")

VOLUME_SHAPE = (200, 200, 512)  # [H, W, D]
DTYPE = np.uint16
OFFSET = 48

EPS = 1e-8

# Pearson correlation 계산 시 너무 많은 voxel을 모두 사용하지 않고
# 일정 간격으로 sampling하여 메모리와 시간을 절약
CORRELATION_SAMPLE_SIZE = 1_000_000
RANDOM_SEED = 42


# ============================================================
# 1. Load .bin volume
# ============================================================

def load_bin_volume(
    file_path: Path,
    shape=VOLUME_SHAPE,
    dtype=DTYPE,
    offset=OFFSET,
):
    """
    Load one .bin file as a 3D volume [H, W, D].
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
# 2. 3D volume Min-Max normalization
# ============================================================

def normalize_volume(volume: np.ndarray):
    """
    Normalize the entire 3D volume to [0, 1].
    """

    volume = volume.astype(np.float32)

    v_min = float(volume.min())
    v_max = float(volume.max())

    scale = v_max - v_min

    if scale < EPS:
        return np.zeros_like(volume, dtype=np.float32)

    return (volume - v_min) / scale


# ============================================================
# 3. Calculate raw volume statistics
# ============================================================

def calculate_volume_statistics(
    volume: np.ndarray,
    file_name: str,
    split: str,
    quality: str,
):
    """
    Calculate raw and normalized statistics for one 3D volume.
    """

    raw = volume.astype(np.float32)
    norm = normalize_volume(volume)

    stats = {
        "split": split,
        "quality": quality,
        "file_name": file_name,

        # Raw statistics
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std()),
        "raw_median": float(np.median(raw)),
        "raw_p90": float(np.percentile(raw, 90)),
        "raw_p95": float(np.percentile(raw, 95)),
        "raw_p99": float(np.percentile(raw, 99)),
        "raw_p99_5": float(np.percentile(raw, 99.5)),
        "raw_p99_9": float(np.percentile(raw, 99.9)),
        "zero_ratio": float(np.mean(raw == 0)),

        # Normalized statistics
        "norm_min": float(norm.min()),
        "norm_max": float(norm.max()),
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std()),
        "norm_median": float(np.median(norm)),
        "norm_p95": float(np.percentile(norm, 95)),
        "norm_p99": float(np.percentile(norm, 99)),
        "norm_p99_9": float(np.percentile(norm, 99.9)),
    }

    return stats


# ============================================================
# 4. Sampled Pearson correlation
# ============================================================

def calculate_sampled_correlation(
    low_volume: np.ndarray,
    high_volume: np.ndarray,
    sample_size: int = CORRELATION_SAMPLE_SIZE,
):
    """
    Calculate Pearson correlation between paired LOW/HIGH volumes.

    Randomly samples voxels if the volume contains more voxels
    than sample_size.
    """

    low_flat = low_volume.reshape(-1).astype(np.float32)
    high_flat = high_volume.reshape(-1).astype(np.float32)

    if low_flat.size != high_flat.size:
        raise ValueError(
            "LOW and HIGH volume sizes do not match."
        )

    total_voxels = low_flat.size

    if total_voxels > sample_size:
        rng = np.random.default_rng(RANDOM_SEED)

        indices = rng.choice(
            total_voxels,
            size=sample_size,
            replace=False,
        )

        low_sample = low_flat[indices]
        high_sample = high_flat[indices]

    else:
        low_sample = low_flat
        high_sample = high_flat

    low_std = low_sample.std()
    high_std = high_sample.std()

    if low_std < EPS or high_std < EPS:
        return np.nan

    correlation = np.corrcoef(
        low_sample,
        high_sample,
    )[0, 1]

    return float(correlation)


# ============================================================
# 5. Calculate pair statistics
# ============================================================

def calculate_pair_statistics(
    low_volume: np.ndarray,
    high_volume: np.ndarray,
    low_file: Path,
    high_file: Path,
    split: str,
):
    """
    Compare one paired LOW/HIGH 3D volume.
    """

    low_raw = low_volume.astype(np.float32)
    high_raw = high_volume.astype(np.float32)

    low_norm = normalize_volume(low_volume)
    high_norm = normalize_volume(high_volume)

    low_p99 = float(np.percentile(low_raw, 99))
    high_p99 = float(np.percentile(high_raw, 99))

    raw_correlation = calculate_sampled_correlation(
        low_raw,
        high_raw,
    )

    norm_correlation = calculate_sampled_correlation(
        low_norm,
        high_norm,
    )

    stats = {
        "split": split,

        "low_file": low_file.name,
        "high_file": high_file.name,

        # Raw LOW
        "low_raw_min": float(low_raw.min()),
        "low_raw_max": float(low_raw.max()),
        "low_raw_mean": float(low_raw.mean()),
        "low_raw_p99": low_p99,

        # Raw HIGH
        "high_raw_min": float(high_raw.min()),
        "high_raw_max": float(high_raw.max()),
        "high_raw_mean": float(high_raw.mean()),
        "high_raw_p99": high_p99,

        # HIGH / LOW intensity ratios
        "max_ratio_high_over_low": (
            float(high_raw.max())
            / (float(low_raw.max()) + EPS)
        ),

        "mean_ratio_high_over_low": (
            float(high_raw.mean())
            / (float(low_raw.mean()) + EPS)
        ),

        "p99_ratio_high_over_low": (
            high_p99
            / (low_p99 + EPS)
        ),

        # Independently normalized mean
        "low_norm_mean": float(low_norm.mean()),
        "high_norm_mean": float(high_norm.mean()),

        # Structural correspondence
        "raw_pearson_correlation": raw_correlation,
        "norm_pearson_correlation": norm_correlation,

        # Difference after independent normalization
        "normalized_mae": float(
            np.mean(np.abs(low_norm - high_norm))
        ),

        "normalized_rmse": float(
            np.sqrt(
                np.mean(
                    (low_norm - high_norm) ** 2
                )
            )
        ),
    }

    return stats


# ============================================================
# 6. Find paired files
# ============================================================

def get_paired_files(
    data_root: Path,
    split: str,
):
    """
    Match files such as:

    Train_000_LOW.bin
    Train_000_HIGH.bin
    """

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

    print(f"\n[{split}]")
    print(f"HIGH files: {len(high_files)}")
    print(f"LOW files : {len(low_files)}")

    if len(high_files) != len(low_files):
        raise ValueError(
            f"Number of HIGH/LOW files does not match.\n"
            f"HIGH: {len(high_files)}\n"
            f"LOW : {len(low_files)}"
        )

    return list(zip(low_files, high_files))


# ============================================================
# 7. Analyze one split
# ============================================================

def analyze_split(
    data_root: Path,
    split: str,
):
    pairs = get_paired_files(
        data_root=data_root,
        split=split,
    )

    volume_statistics = []
    pair_statistics = []

    total = len(pairs)

    for index, (low_file, high_file) in enumerate(
        pairs,
        start=1,
    ):
        print(
            f"[{split}] "
            f"{index}/{total} | "
            f"{low_file.name} <-> {high_file.name}"
        )

        # Load
        low_volume = load_bin_volume(low_file)
        high_volume = load_bin_volume(high_file)

        # Individual volume statistics
        low_stats = calculate_volume_statistics(
            volume=low_volume,
            file_name=low_file.name,
            split=split,
            quality="LOW",
        )

        high_stats = calculate_volume_statistics(
            volume=high_volume,
            file_name=high_file.name,
            split=split,
            quality="HIGH",
        )

        volume_statistics.append(low_stats)
        volume_statistics.append(high_stats)

        # Pair-wise comparison
        pair_stats = calculate_pair_statistics(
            low_volume=low_volume,
            high_volume=high_volume,
            low_file=low_file,
            high_file=high_file,
            split=split,
        )

        pair_statistics.append(pair_stats)

    return (
        pd.DataFrame(volume_statistics),
        pd.DataFrame(pair_statistics),
    )


# ============================================================
# 8. Print summary
# ============================================================

def print_split_summary(
    volume_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    split: str,
):
    print("\n" + "=" * 70)
    print(f"{split} DATA SUMMARY")
    print("=" * 70)

    for quality in ["LOW", "HIGH"]:

        subset = volume_df[
            volume_df["quality"] == quality
        ]

        print(f"\n[{quality}]")

        print(
            f"Raw max mean       : "
            f"{subset['raw_max'].mean():.4f}"
        )

        print(
            f"Raw mean mean      : "
            f"{subset['raw_mean'].mean():.4f}"
        )

        print(
            f"Raw P99 mean       : "
            f"{subset['raw_p99'].mean():.4f}"
        )

        print(
            f"Normalized mean    : "
            f"{subset['norm_mean'].mean():.6f}"
        )

    print("\n[PAIR COMPARISON]")

    print(
        f"Mean HIGH/LOW max ratio : "
        f"{pair_df['max_ratio_high_over_low'].mean():.4f}"
    )

    print(
        f"Mean HIGH/LOW mean ratio: "
        f"{pair_df['mean_ratio_high_over_low'].mean():.4f}"
    )

    print(
        f"Mean HIGH/LOW P99 ratio : "
        f"{pair_df['p99_ratio_high_over_low'].mean():.4f}"
    )

    print(
        f"Mean raw correlation    : "
        f"{pair_df['raw_pearson_correlation'].mean():.4f}"
    )

    print(
        f"Mean normalized corr.   : "
        f"{pair_df['norm_pearson_correlation'].mean():.4f}"
    )

    print(
        f"Mean normalized MAE     : "
        f"{pair_df['normalized_mae'].mean():.6f}"
    )

    print(
        f"Mean normalized RMSE    : "
        f"{pair_df['normalized_rmse'].mean():.6f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in ["Train", "Test"]:

        volume_df, pair_df = analyze_split(
            data_root=DATA_ROOT,
            split=split,
        )

        volume_output_path = (
            OUTPUT_DIR
            / f"{split}_volume_statistics.csv"
        )

        pair_output_path = (
            OUTPUT_DIR
            / f"{split}_pair_statistics.csv"
        )

        volume_df.to_csv(
            volume_output_path,
            index=False,
        )

        pair_df.to_csv(
            pair_output_path,
            index=False,
        )

        print_split_summary(
            volume_df=volume_df,
            pair_df=pair_df,
            split=split,
        )

        print(
            f"\nSaved: {volume_output_path}"
        )

        print(
            f"Saved: {pair_output_path}"
        )


if __name__ == "__main__":
    main()