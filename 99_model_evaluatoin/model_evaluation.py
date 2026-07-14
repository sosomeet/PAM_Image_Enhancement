# -*- coding: utf-8 -*-
"""
PAM multi-model evaluation for the current project structure.

Current project layout
----------------------
PAM_Image_Enhancement/
├─ data/
│  ├─ 3d_Test/
│  │  └─ LOW/        # float32, [X, C, Y, Z] = [200, 3, 200, 512]
│  └─ norm_Test/
│     └─ HIGH/       # float32, [X, Y, Z] = [200, 200, 512]
├─ 11_Dual-Path_AFF_BEFD_SFA_FD-U-Net/
├─ 12_Residual_Edge_Wavelet_MultiStage/
└─ 99_model_evaluatoin/
   └─ model_evaluation.py

Important
---------
- Only experiment folders 11 and 12 are evaluated.
- LOW test data is read from data/3d_Test/LOW.
- HIGH target data is read from data/norm_Test/HIGH.
- LOW files are already pre-generated 3-adjacent Y-Z inputs.
  Therefore, this evaluator DOES NOT create X-1/X/X+1 channels again.
- Y-Z MAP is created by reconstructing prediction [X,Y,Z]
  and applying np.max(volume, axis=0), producing [Y,Z].
- For each model, PNG images are generated only for the test volume with
  the highest SSIM and the test volume with the lowest SSIM.
- Both MAP and slice figures are displayed after a 90-degree clockwise rotation.
- Panel order is LOW -> Prediction -> HIGH -> Absolute Error.
"""

from __future__ import annotations

EVALUATOR_VERSION = "2026-07-14-v4-folders-11-12-only-ssim-extremes-cw90"


import argparse
import csv
import importlib.util
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# =============================================================================
# 1. Paths and data specification
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

TEST_LOW_DIR = DATA_ROOT / "3d_Test" / "LOW"
TEST_HIGH_DIR = DATA_ROOT / "norm_Test" / "HIGH"

RESULT_ROOT = SCRIPT_DIR / "evaluation_results"

HEADER_SIZE = 48
DATA_DTYPE = np.float32

# Pre-generated 3-adjacent Y-Z LOW input:
# [X, C, Y, Z]
ADJACENT_LOW_SHAPE = (200, 3, 200, 512)

# Normalized HIGH target:
# [X, Y, Z]
HIGH_VOLUME_SHAPE = (200, 200, 512)

X_SIZE, Y_SIZE, Z_SIZE = HIGH_VOLUME_SHAPE
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

INPUT_SAMPLE_SHAPE = (INPUT_CHANNELS, Y_SIZE, Z_SIZE)
TARGET_SAMPLE_SHAPE = (OUTPUT_CHANNELS, Y_SIZE, Z_SIZE)

# The normalized test set is evaluated with fixed data range 1.0.
# Prediction is NOT clamped because all training models use linear output.
PSNR_SSIM_DATA_RANGE = 1.0


# =============================================================================
# 2. Model registry
# =============================================================================
#
# 05_Vanilla_U-Net is deliberately not included.
#
# model_file:
#   Exact file name when known. None means recursive auto-discovery.
#
# class_candidates:
#   Exact likely network class names. If none match, the evaluator uses
#   a scored automatic selector that prefers complete U-Net/model classes
#   and rejects helper blocks/layers/attention modules.
# =============================================================================

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "dual_aff_befd_sfa_fd": {
        "display_name": "Dual-Path + AFF + BEFD + SFA + Fully Dense U-Net",
        "folder": "11_Dual-Path_AFF_BEFD_SFA_FD-U-Net",

        # None = automatically discover the model .py file recursively.
        # This is safer than hard-coding the file name.
        "model_file": None,

        "class_candidates": [
            "DualPathAFFBEFDSFAFDUNet",
            "DualPathAFFFullyDenseSFABEFDUNet",
            "DualPathBEFDSFAFDUNet",
            "BEFDSFAFDUNet",
            "FullyDenseSFAUNet",
            "UNet",
            "Model",
        ],
        "init_kwargs": {},
    },

    "residual_edge_wavelet_multistage": {
        "display_name": "Residual + Edge + Wavelet + Multi-Stage U-Net",
        "folder": "12_Residual_Edge_Wavelet_MultiStage",

        # None = automatically discover the model .py file recursively.
        "model_file": None,

        "class_candidates": [
            "ResidualEdgeWaveletMultiStageUNet",
            "ResidualEdgeWaveletMultiStageNet",
            "ResidualEdgeWaveletUNet",
            "ResidualMultiStageUNet",
            "ResidualUNet",
            "UNet",
            "Model",
        ],
        "init_kwargs": {},
    },
}


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    num_workers: int = 0
    use_amp: bool = True
    use_channels_last: bool = True
    max_volumes: Optional[int] = None
    save_slice_examples: int = 3
    save_prediction_volume: bool = False
    strict_checkpoint: bool = True


# =============================================================================
# 3. File utilities
# =============================================================================


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def extract_volume_key(path: Path) -> str:
    """
    Test_000_LOW.bin  -> Test_000
    Test_000_HIGH.bin -> Test_000
    """
    return re.sub(
        r"_(LOW|HIGH)$",
        "",
        path.stem,
        flags=re.IGNORECASE,
    )


def get_bin_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = sorted(directory.glob("*.bin"), key=natural_key)

    if not files:
        raise RuntimeError(f"No .bin files found in: {directory}")

    return files


def validate_bin_file(
    file_path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype = DATA_DTYPE,
    offset: int = HEADER_SIZE,
) -> None:
    dtype = np.dtype(dtype)
    expected_elements = int(np.prod(shape))
    expected_bytes = offset + expected_elements * dtype.itemsize
    actual_bytes = file_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise ValueError(
            "\nBinary file size mismatch\n"
            f"File           : {file_path}\n"
            f"Actual bytes   : {actual_bytes:,}\n"
            f"Expected bytes : {expected_bytes:,}\n"
            f"Expected shape : {shape}\n"
            f"Expected dtype : {dtype}\n"
            f"Header bytes   : {offset}"
        )


def build_test_pairs() -> list[dict[str, Any]]:
    low_files = get_bin_files(TEST_LOW_DIR)
    high_files = get_bin_files(TEST_HIGH_DIR)

    low_map = {extract_volume_key(path): path for path in low_files}
    high_map = {extract_volume_key(path): path for path in high_files}

    common_keys = sorted(low_map.keys() & high_map.keys())

    if not common_keys:
        raise RuntimeError(
            "No matching LOW/HIGH test pairs found.\n"
            f"LOW directory : {TEST_LOW_DIR}\n"
            f"HIGH directory: {TEST_HIGH_DIR}"
        )

    missing_high = sorted(low_map.keys() - high_map.keys())
    missing_low = sorted(high_map.keys() - low_map.keys())

    if missing_high:
        print(
            f"[WARNING] {len(missing_high)} LOW files have no matching HIGH file."
        )
    if missing_low:
        print(
            f"[WARNING] {len(missing_low)} HIGH files have no matching LOW file."
        )

    pairs: list[dict[str, Any]] = []

    for key in common_keys:
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

    return pairs


# =============================================================================
# 4. Dataset
# =============================================================================


class PAMAdjacentYZTestDataset(Dataset):
    """
    One test volume at a time.

    LOW file:
        [X, C, Y, Z] = [200, 3, 200, 512]

    HIGH file:
        [X, Y, Z] = [200, 200, 512]

    One sample at x_index:
        input  = LOW[x_index]      -> [3, 200, 512]
        target = HIGH[x_index]     -> [1, 200, 512]

    The LOW file is already 3-adjacent data.
    No additional neighboring-slice construction is performed.
    """

    def __init__(self, low_path: Path, high_path: Path) -> None:
        super().__init__()

        self.low_path = Path(low_path)
        self.high_path = Path(high_path)

        validate_bin_file(self.low_path, ADJACENT_LOW_SHAPE)
        validate_bin_file(self.high_path, HIGH_VOLUME_SHAPE)

        self.low_volume = np.memmap(
            filename=self.low_path,
            dtype=DATA_DTYPE,
            mode="r",
            offset=HEADER_SIZE,
            shape=ADJACENT_LOW_SHAPE,
            order="C",
        )

        self.high_volume = np.memmap(
            filename=self.high_path,
            dtype=DATA_DTYPE,
            mode="r",
            offset=HEADER_SIZE,
            shape=HIGH_VOLUME_SHAPE,
            order="C",
        )

    def __len__(self) -> int:
        return X_SIZE

    def __getitem__(self, x_index: int) -> dict[str, Any]:
        input_array = np.array(
            self.low_volume[x_index, :, :, :],
            dtype=np.float32,
            copy=True,
        )

        target_array = np.array(
            self.high_volume[x_index, :, :],
            dtype=np.float32,
            copy=True,
        )[None, ...]

        if input_array.shape != INPUT_SAMPLE_SHAPE:
            raise ValueError(
                f"Input shape mismatch at X={x_index}: "
                f"{input_array.shape} != {INPUT_SAMPLE_SHAPE}"
            )

        if target_array.shape != TARGET_SAMPLE_SHAPE:
            raise ValueError(
                f"Target shape mismatch at X={x_index}: "
                f"{target_array.shape} != {TARGET_SAMPLE_SHAPE}"
            )

        return {
            "input": torch.from_numpy(input_array),
            "target": torch.from_numpy(target_array),
            "x_index": x_index,
        }

    def get_low_center_volume(self) -> np.ndarray:
        """
        Returns the central channel as [X,Y,Z].
        Channel 1 is the current X-position Y-Z plane.
        """
        return np.asarray(self.low_volume[:, 1, :, :], dtype=np.float32)

    def get_high_volume(self) -> np.ndarray:
        return np.asarray(self.high_volume, dtype=np.float32)


# =============================================================================
# 5. Dynamic model loading
# =============================================================================


EXCLUDED_PYTHON_FILES = {
    "__init__.py",
    "model_evaluation.py",
    "data_reader.py",
    "dataset.py",
    "utils.py",
    "test.py",
}


def discover_model_file(
    experiment_dir: Path,
    configured_name: Optional[str],
) -> Path:
    if configured_name is not None:
        exact_path = experiment_dir / configured_name
        if exact_path.exists():
            return exact_path

        # Friendly fallback: tolerate small filename spelling differences.
        target_normalized = re.sub(r"[^a-z0-9]", "", configured_name.lower())
        candidates = list(experiment_dir.rglob("*.py"))
        for candidate in candidates:
            normalized = re.sub(r"[^a-z0-9]", "", candidate.name.lower())
            if normalized == target_normalized:
                print(
                    f"[INFO] File-name fallback matched: {candidate.name}"
                )
                return candidate

        raise FileNotFoundError(
            f"Model Python file not found: {exact_path}"
        )

    candidates = [
        path
        for path in experiment_dir.rglob("*.py")
        if path.name.lower() not in EXCLUDED_PYTHON_FILES
        and "__pycache__" not in path.parts
        and ".venv" not in path.parts
    ]

    candidates = sorted(candidates, key=natural_key)

    if not candidates:
        raise FileNotFoundError(
            f"No model Python file found under: {experiment_dir}"
        )

    if len(candidates) > 1:
        print(
            f"[WARNING] Multiple Python files found in {experiment_dir.name}. "
            f"Using: {candidates[0].name}"
        )

    return candidates[0]


def import_module_from_path(module_path: Path, registry_key: str):
    module_name = f"pam_eval_{registry_key}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)

    # Support local sibling imports inside a model folder.
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:
            pass

    return module


def is_model_class(obj: Any) -> bool:
    return (
        inspect.isclass(obj)
        and issubclass(obj, nn.Module)
        and obj is not nn.Module
    )


def architecture_score(cls: type[nn.Module]) -> int:
    name = cls.__name__.lower()
    score = 0

    positive_tokens = {
        "unet": 200,
        "enhancer": 120,
        "network": 100,
        "model": 80,
        "dense": 20,
        "dual": 20,
        "befd": 20,
        "sfa": 20,
        "aff": 20,
        "residual": 30,
        "edge": 20,
        "wavelet": 30,
        "multistage": 30,
        "multi_stage": 30,
    }

    negative_tokens = {
        "block": 250,
        "layer": 250,
        "selector": 250,
        "aggregation": 220,
        "attention": 150,
        "fusion": 150,
        "encoder": 120,
        "decoder": 120,
        "path": 80,
        "branch": 80,
        "conv": 60,
    }

    for token, weight in positive_tokens.items():
        if token in name:
            score += weight

    for token, penalty in negative_tokens.items():
        if token in name:
            score -= penalty

    return score


def select_model_class(
    module: Any,
    class_candidates: Iterable[str],
) -> type[nn.Module]:
    for class_name in class_candidates:
        obj = getattr(module, class_name, None)
        if is_model_class(obj):
            print(f"[INFO] Model class: {class_name}")
            return obj

    discovered = [
        obj
        for _, obj in inspect.getmembers(module, is_model_class)
        if obj.__module__ == module.__name__
    ]

    if not discovered:
        raise RuntimeError("No nn.Module class was found in the model file.")

    ranked = sorted(
        discovered,
        key=lambda cls: (architecture_score(cls), cls.__name__),
        reverse=True,
    )

    best = ranked[0]
    best_score = architecture_score(best)

    if best_score <= 0:
        raise RuntimeError(
            "Model class could not be selected automatically.\n"
            f"Discovered classes: {[cls.__name__ for cls in discovered]}"
        )

    # Avoid ambiguous auto-selection when top scores tie.
    tied = [
        cls for cls in ranked
        if architecture_score(cls) == best_score
    ]

    if len(tied) > 1:
        raise RuntimeError(
            "Ambiguous model classes. Add the exact class name to MODEL_REGISTRY.\n"
            f"Top candidates: {[cls.__name__ for cls in tied]}"
        )

    print(f"[INFO] Auto-selected model class: {best.__name__}")
    return best


def discover_final_checkpoint(experiment_dir: Path) -> Path:
    exact_candidates = [
        experiment_dir / "models_enhancement" / "final" / "model_final.pth",
        experiment_dir / "models_enhancement" / "final" / "model_final.pt",
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    # Fallback: any final checkpoint under the experiment folder.
    patterns = [
        "**/final/model_final.pth",
        "**/final/model_final.pt",
        "**/*final*.pth",
        "**/*final*.pt",
    ]

    discovered: list[Path] = []
    for pattern in patterns:
        discovered.extend(experiment_dir.glob(pattern))

    discovered = sorted(
        {path for path in discovered if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not discovered:
        raise FileNotFoundError(
            f"No final checkpoint found under: {experiment_dir}"
        )

    print(f"[WARNING] Auto-selected checkpoint: {discovered[0]}")
    return discovered[0]


def safe_torch_load(path: Path) -> Any:
    """Load a user-owned checkpoint without the PyTorch FutureWarning."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Older PyTorch without weights_only.
        return torch.load(path, map_location="cpu")
    except Exception as first_error:
        print(
            "[WARNING] weights_only=True failed. "
            "Retrying with weights_only=False for this local checkpoint."
        )
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            raise first_error


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}"
        )

    candidate_keys = (
        "model_state_dict",
        "state_dict",
        "model",
        "network",
        "net",
        "generator",
    )

    for key in candidate_keys:
        value = checkpoint.get(key)

        if isinstance(value, nn.Module):
            return value.state_dict()

        if isinstance(value, dict) and value:
            if all(isinstance(v, torch.Tensor) for v in value.values()):
                return value

    if checkpoint and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in checkpoint.items()
    ):
        return checkpoint

    raise KeyError(
        "No state_dict found in checkpoint. "
        f"Available keys: {list(checkpoint.keys())[:30]}"
    )


def clean_state_dict_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "_orig_mod.",
        "model.",
        "network.",
        "net.",
    )

    cleaned = dict(state_dict)

    changed = True
    while changed:
        changed = False
        keys = list(cleaned.keys())

        for prefix in prefixes:
            if keys and all(key.startswith(prefix) for key in keys):
                cleaned = {
                    key[len(prefix):]: value
                    for key, value in cleaned.items()
                }
                changed = True
                break

    return cleaned


def build_and_load_model(
    registry_key: str,
    device: torch.device,
    strict: bool,
    use_channels_last: bool,
) -> tuple[nn.Module, Path, Path, str]:
    entry = MODEL_REGISTRY[registry_key]
    experiment_dir = PROJECT_ROOT / entry["folder"]

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment folder not found: {experiment_dir}"
        )

    model_file = discover_model_file(
        experiment_dir,
        entry.get("model_file"),
    )

    checkpoint_path = discover_final_checkpoint(experiment_dir)

    module = import_module_from_path(model_file, registry_key)

    model_class = select_model_class(
        module,
        entry["class_candidates"],
    )

    try:
        model = model_class(**entry.get("init_kwargs", {}))
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {model_class.__name__}.\n"
            f"Constructor signature: {inspect.signature(model_class)}\n"
            f"Current init_kwargs: {entry.get('init_kwargs', {})}"
        ) from exc

    checkpoint = safe_torch_load(checkpoint_path)
    state_dict = clean_state_dict_prefixes(extract_state_dict(checkpoint))

    load_result = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if load_result.missing_keys:
            print(
                f"[WARNING] Missing keys: {load_result.missing_keys[:20]}"
            )
        if load_result.unexpected_keys:
            print(
                f"[WARNING] Unexpected keys: "
                f"{load_result.unexpected_keys[:20]}"
            )

    model = model.to(device)

    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model.eval()

    return model, model_file, checkpoint_path, model_class.__name__


def check_model_availability(registry_key: str) -> tuple[bool, str]:
    entry = MODEL_REGISTRY[registry_key]
    experiment_dir = PROJECT_ROOT / entry["folder"]

    if not experiment_dir.exists():
        return False, f"folder missing: {experiment_dir.name}"

    try:
        discover_model_file(experiment_dir, entry.get("model_file"))
    except Exception as exc:
        return False, str(exc)

    try:
        discover_final_checkpoint(experiment_dir)
    except Exception as exc:
        return False, str(exc)

    return True, "available"


# =============================================================================
# 6. Prediction handling and metrics
# =============================================================================


def extract_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        prediction = output

    elif isinstance(output, (tuple, list)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        if not tensors:
            raise TypeError("Model output tuple/list contains no tensor.")
        prediction = tensors[0]

    elif isinstance(output, dict):
        preferred_keys = (
            "prediction",
            "pred",
            "output",
            "out",
            "enhanced",
            "final",
        )

        prediction = None
        for key in preferred_keys:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                prediction = value
                break

        if prediction is None:
            prediction = next(
                (
                    value
                    for value in output.values()
                    if isinstance(value, torch.Tensor)
                ),
                None,
            )

        if prediction is None:
            raise TypeError("Model output dictionary contains no tensor.")

    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)

    if prediction.ndim != 4:
        raise ValueError(
            f"Expected prediction [B,C,Y,Z], got {tuple(prediction.shape)}"
        )

    if prediction.shape[1] != 1:
        raise ValueError(
            f"Expected one output channel, got {prediction.shape[1]}"
        )

    return prediction


def gaussian_kernel(
    window_size: int,
    sigma: float,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.arange(
        window_size,
        dtype=dtype,
        device=device,
    )
    coordinates -= window_size // 2

    gaussian = torch.exp(
        -(coordinates ** 2) / (2 * sigma ** 2)
    )
    gaussian /= gaussian.sum()

    kernel_2d = gaussian[:, None] @ gaussian[None, :]

    return kernel_2d.expand(
        channels,
        1,
        window_size,
        window_size,
    ).contiguous()


def calculate_ssim_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = PSNR_SSIM_DATA_RANGE,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM shape mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )

    channels = prediction.shape[1]
    kernel = gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        dtype=prediction.dtype,
        device=prediction.device,
    )

    padding = window_size // 2

    mu_x = F.conv2d(
        prediction,
        kernel,
        padding=padding,
        groups=channels,
    )
    mu_y = F.conv2d(
        target,
        kernel,
        padding=padding,
        groups=channels,
    )

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(
        prediction * prediction,
        kernel,
        padding=padding,
        groups=channels,
    ) - mu_x_sq

    sigma_y_sq = F.conv2d(
        target * target,
        kernel,
        padding=padding,
        groups=channels,
    ) - mu_y_sq

    sigma_xy = F.conv2d(
        prediction * target,
        kernel,
        padding=padding,
        groups=channels,
    ) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (
        (mu_x_sq + mu_y_sq + c1)
        * (sigma_x_sq + sigma_y_sq + c2)
    )

    ssim_map = numerator / torch.clamp(
        denominator,
        min=torch.finfo(prediction.dtype).eps,
    )

    return ssim_map.flatten(1).mean(dim=1)


def calculate_psnr_from_mse(
    mse: float,
    data_range: float = PSNR_SSIM_DATA_RANGE,
) -> float:
    if mse <= 0:
        return float("inf")

    return 10.0 * math.log10((data_range ** 2) / mse)


# =============================================================================
# 7. Visualization
# =============================================================================


def robust_limits(
    images: Iterable[np.ndarray],
    lower_percentile: float = 0.0,
    upper_percentile: float = 99.8,
) -> tuple[float, float]:
    values = np.concatenate([image.ravel() for image in images])

    v_min = float(np.percentile(values, lower_percentile))
    v_max = float(np.percentile(values, upper_percentile))

    if v_max <= v_min:
        v_max = v_min + 1e-8

    return v_min, v_max


def save_yz_map_figure(
    low_center_volume: np.ndarray,
    high_volume: np.ndarray,
    prediction_volume: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    """
    All input volumes are [X,Y,Z].

    Y-Z MAP:
        max projection along X
        np.max(volume, axis=0)
        -> [Y,Z]

    Display:
        Every MAP image is rotated 90 degrees clockwise.
        np.rot90(image, k=-1)

    Panel order:
        LOW -> Prediction -> HIGH -> Absolute Error
    """
    low_map = np.max(low_center_volume, axis=0)
    high_map = np.max(high_volume, axis=0)
    prediction_map = np.max(prediction_volume, axis=0)
    error_map = np.abs(high_map - prediction_map)

    # Rotate every MAP 90 degrees clockwise for display.
    low_map = np.rot90(low_map, k=-1)
    prediction_map = np.rot90(prediction_map, k=-1)
    high_map = np.rot90(high_map, k=-1)
    error_map = np.rot90(error_map, k=-1)

    v_min, v_max = robust_limits(
        [low_map, prediction_map, high_map]
    )

    error_max = float(np.percentile(error_map, 99.8))
    if error_max <= 0:
        error_max = 1e-8

    fig, axes = plt.subplots(1, 4, figsize=(22, 10))

    panels = [
        (low_map, "LOW Y-Z MAP", "hot", v_min, v_max),
        (
            prediction_map,
            "Prediction Y-Z MAP",
            "hot",
            v_min,
            v_max,
        ),
        (high_map, "HIGH Y-Z MAP", "hot", v_min, v_max),
        (
            error_map,
            "Absolute Error Y-Z MAP",
            "viridis",
            0.0,
            error_max,
        ),
    ]

    for axis, (image, panel_title, cmap, p_min, p_max) in zip(axes, panels):
        rendered = axis.imshow(
            image,
            cmap=cmap,
            origin="upper",
            aspect="auto",
            vmin=p_min,
            vmax=p_max,
        )
        axis.set_title(panel_title)
        # Original [Y,Z] becomes [Z,Y] after clockwise rotation.
        # Therefore the displayed horizontal axis corresponds to Y,
        # and the displayed vertical axis corresponds to Z.
        axis.set_xlabel("Y")
        axis.set_ylabel("Z")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title + " | MAP rotated 90° clockwise")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_yz_slice_figure(
    low_center_volume: np.ndarray,
    high_volume: np.ndarray,
    prediction_volume: np.ndarray,
    x_index: int,
    save_path: Path,
) -> None:
    """
    Save one Y-Z slice at a fixed X index.

    Display:
        Every slice image is rotated 90 degrees clockwise.
        np.rot90(image, k=-1)

    Panel order:
        LOW -> Prediction -> HIGH -> Absolute Error
    """
    low_slice = low_center_volume[x_index]
    high_slice = high_volume[x_index]
    prediction_slice = prediction_volume[x_index]
    error_slice = np.abs(high_slice - prediction_slice)

    # Rotate every Y-Z slice 90 degrees clockwise for display.
    low_slice = np.rot90(low_slice, k=-1)
    prediction_slice = np.rot90(prediction_slice, k=-1)
    high_slice = np.rot90(high_slice, k=-1)
    error_slice = np.rot90(error_slice, k=-1)

    v_min, v_max = robust_limits(
        [low_slice, prediction_slice, high_slice]
    )

    error_max = float(np.percentile(error_slice, 99.8))
    if error_max <= 0:
        error_max = 1e-8

    fig, axes = plt.subplots(1, 4, figsize=(22, 10))

    panels = [
        (low_slice, "LOW", "hot", v_min, v_max),
        (prediction_slice, "Prediction", "hot", v_min, v_max),
        (high_slice, "HIGH", "hot", v_min, v_max),
        (error_slice, "Absolute Error", "viridis", 0.0, error_max),
    ]

    for axis, (image, panel_title, cmap, p_min, p_max) in zip(axes, panels):
        rendered = axis.imshow(
            image,
            cmap=cmap,
            origin="upper",
            aspect="auto",
            vmin=p_min,
            vmax=p_max,
        )
        axis.set_title(panel_title)
        # Original [Y,Z] becomes [Z,Y] after clockwise rotation.
        axis.set_xlabel("Y")
        axis.set_ylabel("Z")
        fig.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Y-Z slice at X={x_index} | rotated 90° clockwise"
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# 8. Evaluation
# =============================================================================


def evaluate_single_volume(
    model: nn.Module,
    pair: dict[str, Any],
    device: torch.device,
    config: EvaluationConfig,
    model_result_dir: Path,
    display_name: str,
    save_visualization: bool,
    visualization_label: Optional[str] = None,
    save_prediction_output: bool = True,
) -> dict[str, Any]:
    """
    Evaluate one [X,Y,Z] test volume.

    Normal evaluation pass:
        save_visualization=False
        -> metrics are calculated without reconstructing a full CPU prediction
           volume unless --save-prediction-volume is enabled.

    SSIM-extreme visualization pass:
        save_visualization=True
        -> reconstruct the full prediction volume and save MAP/slice figures.
    """
    dataset = PAMAdjacentYZTestDataset(
        low_path=pair["low_path"],
        high_path=pair["high_path"],
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }

    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)

    need_prediction_volume = (
        save_visualization
        or (config.save_prediction_volume and save_prediction_output)
    )

    prediction_volume: Optional[np.ndarray]
    if need_prediction_volume:
        prediction_volume = np.empty(
            HIGH_VOLUME_SHAPE,
            dtype=np.float32,
        )
    else:
        prediction_volume = None

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    element_count = 0

    ssim_sum = 0.0
    slice_count = 0

    amp_enabled = config.use_amp and device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"    {pair['volume_key']}",
        leave=False,
        dynamic_ncols=True,
    )

    model.eval()

    with torch.inference_mode():
        for batch in progress:
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

            if config.use_channels_last and device.type == "cuda":
                inputs = inputs.contiguous(
                    memory_format=torch.channels_last
                )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(inputs)
                predictions = extract_prediction(output)

                if predictions.shape[-2:] != targets.shape[-2:]:
                    predictions = F.interpolate(
                        predictions,
                        size=targets.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Do NOT clamp. These models use linear output.
                diff = predictions - targets

                batch_absolute_error = diff.abs().sum()
                batch_squared_error = diff.square().sum()

                batch_ssim = calculate_ssim_per_sample(
                    predictions.float(),
                    targets.float(),
                )

            absolute_error_sum += float(batch_absolute_error.item())
            squared_error_sum += float(batch_squared_error.item())
            element_count += int(targets.numel())

            ssim_sum += float(batch_ssim.sum().item())
            slice_count += int(targets.shape[0])

            if prediction_volume is not None:
                batch_predictions = (
                    predictions[:, 0]
                    .float()
                    .cpu()
                    .numpy()
                )

                x_indices = batch["x_index"]
                if isinstance(x_indices, torch.Tensor):
                    x_indices = x_indices.tolist()

                for local_index, x_index in enumerate(x_indices):
                    prediction_volume[int(x_index)] = (
                        batch_predictions[local_index]
                    )

    l1 = absolute_error_sum / element_count
    mse = squared_error_sum / element_count
    psnr = calculate_psnr_from_mse(mse)
    ssim = ssim_sum / slice_count

    volume_key = pair["volume_key"]

    if save_visualization:
        if prediction_volume is None:
            raise RuntimeError(
                "prediction_volume was not reconstructed for visualization."
            )

        label = visualization_label or "visualization"
        volume_dir = model_result_dir / label / volume_key
        volume_dir.mkdir(parents=True, exist_ok=True)

        low_center_volume = dataset.get_low_center_volume()
        high_volume = dataset.get_high_volume()

        save_yz_map_figure(
            low_center_volume=low_center_volume,
            high_volume=high_volume,
            prediction_volume=prediction_volume,
            save_path=volume_dir / "yz_map_comparison.png",
            title=(
                f"{display_name} | {label} | {volume_key} | "
                f"SSIM={ssim:.6f}"
            ),
        )

        if config.save_slice_examples > 0:
            x_indices = np.linspace(
                0,
                X_SIZE - 1,
                num=config.save_slice_examples,
                dtype=int,
            )

            for x_index in np.unique(x_indices):
                save_yz_slice_figure(
                    low_center_volume=low_center_volume,
                    high_volume=high_volume,
                    prediction_volume=prediction_volume,
                    x_index=int(x_index),
                    save_path=(
                        volume_dir
                        / f"yz_slice_x_{int(x_index):03d}.png"
                    ),
                )

    if (
        config.save_prediction_volume
        and save_prediction_output
        and prediction_volume is not None
    ):
        volume_dir = model_result_dir / volume_key
        volume_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            volume_dir / "prediction_volume.npy",
            prediction_volume,
        )

    result = {
        "volume": volume_key,
        "low_file": str(pair["low_path"]),
        "high_file": str(pair["high_path"]),
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }

    del prediction_volume
    del dataset

    return result

def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    registry_key: str,
    test_pairs: list[dict[str, Any]],
    device: torch.device,
    config: EvaluationConfig,
) -> dict[str, Any]:
    entry = MODEL_REGISTRY[registry_key]

    model, model_file, checkpoint_path, class_name = build_and_load_model(
        registry_key=registry_key,
        device=device,
        strict=config.strict_checkpoint,
        use_channels_last=config.use_channels_last,
    )

    selected_pairs = test_pairs
    if config.max_volumes is not None:
        selected_pairs = selected_pairs[: config.max_volumes]

    model_result_dir = RESULT_ROOT / registry_key
    model_result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"Model          : {entry['display_name']}")
    print(f"Registry key   : {registry_key}")
    print(f"Model class    : {class_name}")
    print(f"Model file     : {model_file}")
    print(f"Final model    : {checkpoint_path}")
    print(f"Test LOW       : {TEST_LOW_DIR}")
    print(f"Test HIGH      : {TEST_HIGH_DIR}")
    print(f"Volumes        : {len(selected_pairs)}")
    print("PNG samples    : highest-SSIM 1 + lowest-SSIM 1 per model")
    print(f"Device         : {device}")
    print("=" * 88)

    per_volume_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pass 1: evaluate every selected test volume and collect metrics.
    # No PNG is generated in this pass.
    # ------------------------------------------------------------------
    for index, pair in enumerate(selected_pairs, start=1):
        print(
            f"[{index:03d}/{len(selected_pairs):03d}] "
            f"{pair['volume_key']}"
        )

        result = evaluate_single_volume(
            model=model,
            pair=pair,
            device=device,
            config=config,
            model_result_dir=model_result_dir,
            display_name=entry["display_name"],
            save_visualization=False,
        )

        result = {
            "model_key": registry_key,
            "model_name": entry["display_name"],
            **result,
        }

        per_volume_rows.append(result)

        print(
            "    "
            f"L1={result['l1']:.8f} | "
            f"MSE={result['mse']:.8f} | "
            f"PSNR={result['psnr']:.4f} dB | "
            f"SSIM={result['ssim']:.6f}"
        )

    if not per_volume_rows:
        raise RuntimeError(
            f"No test volumes were evaluated for model: {registry_key}"
        )

    save_csv(
        per_volume_rows,
        model_result_dir / "per_volume_metrics.csv",
    )

    # ------------------------------------------------------------------
    # Select SSIM extremes for this model.
    # SSIM is the per-volume mean over all X-indexed Y-Z slices.
    # ------------------------------------------------------------------
    highest_ssim_row = max(
        per_volume_rows,
        key=lambda row: float(row["ssim"]),
    )
    lowest_ssim_row = min(
        per_volume_rows,
        key=lambda row: float(row["ssim"]),
    )

    pair_by_key = {
        pair["volume_key"]: pair
        for pair in selected_pairs
    }

    extremes = {
        "highest_ssim": {
            "volume": highest_ssim_row["volume"],
            "ssim": float(highest_ssim_row["ssim"]),
            "l1": float(highest_ssim_row["l1"]),
            "mse": float(highest_ssim_row["mse"]),
            "psnr": float(highest_ssim_row["psnr"]),
        },
        "lowest_ssim": {
            "volume": lowest_ssim_row["volume"],
            "ssim": float(lowest_ssim_row["ssim"]),
            "l1": float(lowest_ssim_row["l1"]),
            "mse": float(lowest_ssim_row["mse"]),
            "psnr": float(lowest_ssim_row["psnr"]),
        },
    }

    with (model_result_dir / "ssim_extremes.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(extremes, file, indent=2, ensure_ascii=False)

    print("\nSSIM extremes selected for PNG generation")
    print(
        "  Highest SSIM : "
        f"{highest_ssim_row['volume']} "
        f"(SSIM={highest_ssim_row['ssim']:.6f})"
    )
    print(
        "  Lowest SSIM  : "
        f"{lowest_ssim_row['volume']} "
        f"(SSIM={lowest_ssim_row['ssim']:.6f})"
    )

    # ------------------------------------------------------------------
    # Pass 2: re-run only the two selected extreme volumes to reconstruct
    # full predictions and save figures.
    # ------------------------------------------------------------------
    visualization_targets = [
        ("highest_ssim", highest_ssim_row),
        ("lowest_ssim", lowest_ssim_row),
    ]

    for label, row in visualization_targets:
        pair = pair_by_key[row["volume"]]

        print(
            f"  [PNG] {label}: {row['volume']} "
            f"(SSIM={row['ssim']:.6f})"
        )

        evaluate_single_volume(
            model=model,
            pair=pair,
            device=device,
            config=config,
            model_result_dir=model_result_dir,
            display_name=entry["display_name"],
            save_visualization=True,
            visualization_label=label,
            # Avoid re-saving prediction_volume.npy during this second pass.
            save_prediction_output=False,
        )

    summary = {
        "model_key": registry_key,
        "model_name": entry["display_name"],
        "model_class": class_name,
        "num_volumes": len(per_volume_rows),
        "mean_l1": float(np.mean([row["l1"] for row in per_volume_rows])),
        "mean_mse": float(np.mean([row["mse"] for row in per_volume_rows])),
        "mean_psnr": float(np.mean([row["psnr"] for row in per_volume_rows])),
        "mean_ssim": float(np.mean([row["ssim"] for row in per_volume_rows])),
        "highest_ssim_volume": highest_ssim_row["volume"],
        "highest_ssim": float(highest_ssim_row["ssim"]),
        "lowest_ssim_volume": lowest_ssim_row["volume"],
        "lowest_ssim": float(lowest_ssim_row["ssim"]),
        "model_file": str(model_file),
        "checkpoint": str(checkpoint_path),
    }

    with (model_result_dir / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("-" * 88)
    print(
        f"Mean | L1={summary['mean_l1']:.8f} | "
        f"MSE={summary['mean_mse']:.8f} | "
        f"PSNR={summary['mean_psnr']:.4f} dB | "
        f"SSIM={summary['mean_ssim']:.6f}"
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary

# =============================================================================
# 9. CLI
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PAM Y-Z 3-adjacent enhancement models "
            "from experiment folders 11 and 12 only."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Model keys to evaluate or 'all'. Available: "
            + ", ".join(MODEL_REGISTRY.keys())
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Evaluate only the first N test volumes.",
    )



    parser.add_argument(
        "--slice-examples",
        type=int,
        default=3,
        help=(
            "Number of representative Y-Z slice figures saved for each "
            "SSIM-extreme volume (highest and lowest)."
        ),
    )

    parser.add_argument(
        "--save-prediction-volume",
        action="store_true",
        help=(
            "Save prediction_volume.npy for every test volume. "
            "Disabled by default to avoid large disk usage."
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--no-channels-last",
        action="store_true",
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Load state_dict with strict=False.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(option: str) -> torch.device:
    if option == "cpu":
        return torch.device("cpu")

    if option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def resolve_model_keys(requested: list[str]) -> list[str]:
    if "all" in requested:
        return list(MODEL_REGISTRY.keys())

    unknown = [
        key for key in requested
        if key not in MODEL_REGISTRY
    ]

    if unknown:
        raise KeyError(
            f"Unknown model keys: {unknown}\n"
            f"Available keys: {list(MODEL_REGISTRY.keys())}"
        )

    return requested


def main() -> None:
    args = parse_arguments()

    print(f"[EVALUATOR VERSION] {EVALUATOR_VERSION}")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.max_volumes is not None and args.max_volumes <= 0:
        raise ValueError("--max-volumes must be positive.")

    if args.slice_examples < 0:
        raise ValueError("--slice-examples cannot be negative.")

    device = resolve_device(args.device)
    requested_model_keys = resolve_model_keys(args.models)
    test_pairs = build_test_pairs()

    config = EvaluationConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        use_channels_last=not args.no_channels_last,
        max_volumes=args.max_volumes,
        save_slice_examples=args.slice_examples,
        save_prediction_volume=args.save_prediction_volume,
        strict_checkpoint=not args.non_strict,
    )

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PAM MODEL EVALUATION — FOLDERS 11 AND 12 ONLY — SSIM EXTREMES")
    print("=" * 88)
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Test LOW         : {TEST_LOW_DIR}")
    print(f"Test HIGH        : {TEST_HIGH_DIR}")
    print(f"Paired volumes   : {len(test_pairs)}")
    print(f"LOW shape        : {ADJACENT_LOW_SHAPE} [X,C,Y,Z]")
    print(f"HIGH shape       : {HIGH_VOLUME_SHAPE} [X,Y,Z]")
    print(f"dtype            : {np.dtype(DATA_DTYPE)}")
    print(f"Header size      : {HEADER_SIZE} bytes")
    print(f"Requested models : {requested_model_keys}")
    print("PNG policy       : highest SSIM 1 + lowest SSIM 1 per model")
    print("Image rotation   : 90 degrees clockwise for MAP and slice")
    print("Panel order      : LOW -> Prediction -> HIGH -> Absolute Error")
    print(f"Device           : {device}")

    if device.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")

    print("=" * 88)

    # Skip folders that do not yet contain both a model Python file and final model.
    available_keys: list[str] = []
    skipped_rows: list[dict[str, str]] = []

    print("\nModel availability")
    for registry_key in requested_model_keys:
        available, reason = check_model_availability(registry_key)

        if available:
            available_keys.append(registry_key)
            print(f"  [READY] {registry_key}")
        else:
            skipped_rows.append(
                {
                    "model_key": registry_key,
                    "reason": reason,
                }
            )
            print(f"  [SKIP ] {registry_key}: {reason}")

    if not available_keys:
        print("\nNo available models were found. Nothing to evaluate.")
        return

    summaries: list[dict[str, Any]] = []
    failure_rows: list[dict[str, str]] = []

    for registry_key in available_keys:
        try:
            summary = evaluate_model(
                registry_key=registry_key,
                test_pairs=test_pairs,
                device=device,
                config=config,
            )
            summaries.append(summary)

        except Exception as exc:
            failure_rows.append(
                {
                    "model_key": registry_key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print("\n" + "!" * 88)
            print(f"[FAILED] {registry_key}")
            print(f"{type(exc).__name__}: {exc}")
            print("!" * 88)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    if summaries:
        comparison_rows = [
            {
                "model_key": item["model_key"],
                "model_name": item["model_name"],
                "model_class": item["model_class"],
                "num_volumes": item["num_volumes"],
                "mean_l1": item["mean_l1"],
                "mean_mse": item["mean_mse"],
                "mean_psnr": item["mean_psnr"],
                "mean_ssim": item["mean_ssim"],
                "highest_ssim_volume": item["highest_ssim_volume"],
                "highest_ssim": item["highest_ssim"],
                "lowest_ssim_volume": item["lowest_ssim_volume"],
                "lowest_ssim": item["lowest_ssim"],
                "checkpoint": item["checkpoint"],
            }
            for item in summaries
        ]

        save_csv(
            comparison_rows,
            RESULT_ROOT / "model_comparison.csv",
        )

    if skipped_rows:
        with (RESULT_ROOT / "skipped_models.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                skipped_rows,
                file,
                indent=2,
                ensure_ascii=False,
            )

    if failure_rows:
        with (RESULT_ROOT / "failed_models.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                failure_rows,
                file,
                indent=2,
                ensure_ascii=False,
            )

    print("\n" + "=" * 88)
    print("EVALUATION FINISHED")
    print(f"Successful models : {len(summaries)}")
    print(f"Skipped models    : {len(skipped_rows)}")
    print(f"Failed models     : {len(failure_rows)}")
    print(f"Results           : {RESULT_ROOT}")
    print("=" * 88)

    if summaries:
        print("\nModel comparison")
        for item in summaries:
            print(
                f"  {item['model_key']:<24} | "
                f"L1={item['mean_l1']:.8f} | "
                f"PSNR={item['mean_psnr']:.4f} dB | "
                f"SSIM={item['mean_ssim']:.6f} | "
                f"Best={item['highest_ssim_volume']} "
                f"({item['highest_ssim']:.6f}) | "
                f"Worst={item['lowest_ssim_volume']} "
                f"({item['lowest_ssim']:.6f})"
            )

if __name__ == "__main__":
    main()
