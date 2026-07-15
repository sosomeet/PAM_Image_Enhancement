# -*- coding: utf-8 -*-

"""
E9 PAM 3D Evaluation

LOW:
    float32 + 48-byte header
    [X, C, Y, Z] = [200, 3, 200, 512]

Prediction:
    E9 model inference
    [X, Y, Z] = [200, 200, 512]

HIGH:
    float32 + 48-byte header
    [X, Y, Z] = [200, 200, 512]

Napari:
    LOW | PREDICTION | HIGH

The three volumes are placed side-by-side.
Object positions, camera center, and zoom remain fixed.
Only 3D camera rotation is allowed.
"""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import napari


# ============================================================
# 0. Basic settings
# ============================================================

# ------------------------------------------------------------
# E9 model definition script
# ------------------------------------------------------------
#
# 현재 E9 학습 코드를 저장한 실제 Python 파일 경로로 수정.
#
# 예:
# "./12_Residual_Edge_Wavelet_MultiStage/Residual_Edge_Wavelet_MultiStage_U-Net.py"
#
# .txt 파일이어도 SourceFileLoader를 사용하므로 로드 가능.
# ------------------------------------------------------------

MODEL_DEFINITION_PATH = Path(
    "./12_Residual_Edge_Wavelet_MultiStage/Residual_Edge_Wavelet_MultiStage_U-Net.py"
)


# ------------------------------------------------------------
# Trained checkpoint
# ------------------------------------------------------------

CHECKPOINT_PATH = Path(
    "./12_Residual_Edge_Wavelet_MultiStage/"
    "models_enhancement/final/model_final.pth"
)


# ------------------------------------------------------------
# Evaluation data
# ------------------------------------------------------------
#
# 중요:
# 현재 E9 모델은 기존 raw LOW가 아니라
# 사전에 생성된 3-adjacent LOW volume을 입력으로 사용.
#
# LOW:
# [X,C,Y,Z] = [200,3,200,512]
#
# HIGH:
# [X,Y,Z] = [200,200,512]
# ------------------------------------------------------------

LOW_PATH = Path(
    "./data/3d_Train/LOW/Train_001_LOW.bin"
)

HIGH_PATH = Path(
    "./data/norm_Train/HIGH/Train_001_HiGH.bin"
)


# ------------------------------------------------------------
# Data format
# ------------------------------------------------------------

LOW_SHAPE = (200, 3, 200, 512)    # [X,C,Y,Z]
HIGH_SHAPE = (200, 200, 512)      # [X,Y,Z]

DTYPE = np.float32
OFFSET = 48

X_SIZE, _, Y_SIZE, Z_SIZE = LOW_SHAPE


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

INFERENCE_BATCH_SIZE = 4

USE_AMP = True
USE_CHANNELS_LAST = True


# ------------------------------------------------------------
# 3D visualization
# ------------------------------------------------------------

GAP = 50

# "independent":
#     LOW / Prediction / HIGH 각각 별도의 contrast.
#     구조를 보기에는 편하지만 실제 intensity 차이를 숨길 수 있음.
#
# "shared":
#     LOW / Prediction / HIGH에 같은 contrast 적용.
#     현재 intensity preservation 비교에는 이것을 권장.
CONTRAST_MODE = "shared"

LOW_PERCENTILE = 0.0
HIGH_PERCENTILE = 99.5

COLORMAP = "hot"
RENDERING = "mip"


# ============================================================
# 1. Dynamically load E9 architecture
# ============================================================

def load_e9_class(
    model_definition_path: Path,
):
    """
    E9 학습 Python 파일에서
    E9ResidualEdgeWaveletMultiStageNet 클래스를 동적으로 가져온다.

    학습 파일 내부의:

        if __name__ == "__main__":
            train()

    은 import 시 실행되지 않으므로 재학습은 시작되지 않는다.
    """

    model_definition_path = Path(model_definition_path)

    if not model_definition_path.exists():
        raise FileNotFoundError(
            "\nE9 model definition file not found.\n"
            f"Path: {model_definition_path}\n\n"
            "MODEL_DEFINITION_PATH를 실제 E9 학습 코드 경로로 "
            "수정하세요."
        )

    loader = SourceFileLoader(
        "e9_training_module",
        str(model_definition_path),
    )

    spec = spec_from_loader(
        loader.name,
        loader,
    )

    if spec is None:
        raise ImportError(
            f"Could not create module spec: {model_definition_path}"
        )

    module = module_from_spec(spec)
    loader.exec_module(module)

    if not hasattr(
        module,
        "E9ResidualEdgeWaveletMultiStageNet",
    ):
        raise AttributeError(
            "\nModel class not found.\n"
            "Expected class:\n"
            "E9ResidualEdgeWaveletMultiStageNet\n\n"
            f"File: {model_definition_path}"
        )

    model_class = getattr(
        module,
        "E9ResidualEdgeWaveletMultiStageNet",
    )

    return model_class


# ============================================================
# 2. Load binary volume
# ============================================================

def load_bin_volume(
    path: Path,
    shape: tuple[int, ...],
    dtype=np.float32,
    offset: int = 48,
) -> np.ndarray:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    raw = np.fromfile(
        path,
        dtype=dtype,
        offset=offset,
    )

    expected_size = int(np.prod(shape))

    print("\n" + "=" * 80)
    print(f"Loading: {path}")
    print("=" * 80)

    print(f"Raw elements     : {raw.size:,}")
    print(f"Expected elements: {expected_size:,}")
    print(f"Expected shape   : {shape}")
    print(f"Dtype            : {np.dtype(dtype)}")
    print(f"Header           : {offset} bytes")

    if raw.size != expected_size:
        raise ValueError(
            "\nData size mismatch\n"
            f"File    : {path}\n"
            f"Raw     : {raw.size:,}\n"
            f"Expected: {expected_size:,}\n"
            f"Shape   : {shape}\n"
            f"Dtype   : {np.dtype(dtype)}"
        )

    volume = raw.reshape(shape)

    print(f"Volume shape     : {volume.shape}")
    print(f"Volume dtype     : {volume.dtype}")
    print(f"Volume min       : {volume.min():.8f}")
    print(f"Volume max       : {volume.max():.8f}")
    print(f"Volume mean      : {volume.mean():.8f}")

    return volume


# ============================================================
# 3. Load trained E9 model
# ============================================================

def load_trained_e9_model(
    model_class,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("\n" + "=" * 80)
    print("LOADING E9 MODEL")
    print("=" * 80)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device    : {device}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # 현재 E9 checkpoint에는 model_metadata()의 결과가 같이 저장됨.
    #
    # 따라서 학습 당시 설정을 checkpoint에서 직접 읽음.
    # --------------------------------------------------------

    base_channels = int(
        checkpoint.get(
            "base_channels",
            16,
        )
    )

    max_channels = int(
        checkpoint.get(
            "max_channels",
            128,
        )
    )

    use_edge_branch = bool(
        checkpoint.get(
            "edge_branch",
            True,
        )
    )

    use_wavelet_branch = bool(
        checkpoint.get(
            "wavelet_branch",
            True,
        )
    )

    use_stage2 = bool(
        checkpoint.get(
            "stage2_refinement",
            True,
        )
    )

    print(f"Architecture     : {checkpoint.get('architecture', 'Unknown')}")
    print(f"Epoch            : {checkpoint.get('epoch', 'Unknown')}")
    print(f"Base channels    : {base_channels}")
    print(f"Max channels     : {max_channels}")
    print(f"Edge branch      : {use_edge_branch}")
    print(f"Wavelet branch   : {use_wavelet_branch}")
    print(f"Stage-2          : {use_stage2}")

    model = model_class(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels,
        max_channels=max_channels,
        use_edge_branch=use_edge_branch,
        use_wavelet_branch=use_wavelet_branch,
        use_stage2=use_stage2,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)

    if USE_CHANNELS_LAST and device.type == "cuda":
        model = model.to(
            memory_format=torch.channels_last
        )

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# 4. Predict complete 3D volume
# ============================================================

def predict_full_volume(
    model: nn.Module,
    low_volume: np.ndarray,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Input:
        low_volume:
            [X,C,Y,Z]
            [200,3,200,512]

    Model prediction:
        each X slice:
            [3,200,512]
                ↓ E9
            [1,200,512]

    Output:
        prediction_volume:
            [X,Y,Z]
            [200,200,512]
    """

    if low_volume.shape != LOW_SHAPE:
        raise ValueError(
            "\nLOW volume shape mismatch\n"
            f"Actual  : {low_volume.shape}\n"
            f"Expected: {LOW_SHAPE}"
        )

    prediction_volume = np.empty(
        HIGH_SHAPE,
        dtype=np.float32,
    )

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    print("\n" + "=" * 80)
    print("E9 FULL 3D VOLUME INFERENCE")
    print("=" * 80)

    print(f"LOW volume       : {low_volume.shape}")
    print(f"Prediction target: {prediction_volume.shape}")
    print(f"Batch size       : {batch_size}")
    print(f"AMP              : {amp_enabled}")
    print(f"Channels last    : {USE_CHANNELS_LAST}")
    print()

    with torch.inference_mode():

        for start in range(
            0,
            X_SIZE,
            batch_size,
        ):
            end = min(
                start + batch_size,
                X_SIZE,
            )

            # -----------------------------------------------
            # [B,3,200,512]
            # -----------------------------------------------

            batch_np = np.array(
                low_volume[start:end],
                dtype=np.float32,
                copy=True,
            )

            batch = torch.from_numpy(
                batch_np
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if (
                USE_CHANNELS_LAST
                and device.type == "cuda"
            ):
                batch = batch.contiguous(
                    memory_format=torch.channels_last
                )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(batch)

            # -----------------------------------------------
            # prediction:
            # [B,1,200,512]
            #
            # prediction[:, 0]:
            # [B,200,512]
            # -----------------------------------------------

            prediction_np = (
                prediction[:, 0]
                .float()
                .cpu()
                .numpy()
            )

            prediction_volume[start:end] = (
                prediction_np
            )

            print(
                f"\rInference: "
                f"{end:3d}/{X_SIZE} X slices",
                end="",
                flush=True,
            )

    print("\n")

    print("Prediction completed.")
    print(f"Shape : {prediction_volume.shape}")
    print(f"Dtype : {prediction_volume.dtype}")
    print(f"Min   : {prediction_volume.min():.8f}")
    print(f"Max   : {prediction_volume.max():.8f}")
    print(f"Mean  : {prediction_volume.mean():.8f}")

    # --------------------------------------------------------
    # E9에는 output activation과 clamping이 없으므로
    # [0,1] 밖의 prediction 비율도 출력.
    # 실제 결과 자체는 수정하지 않음.
    # --------------------------------------------------------

    below_zero = float(
        np.mean(prediction_volume < 0.0) * 100.0
    )

    above_one = float(
        np.mean(prediction_volume > 1.0) * 100.0
    )

    print(
        f"Values < 0 : {below_zero:.6f}%"
    )

    print(
        f"Values > 1 : {above_one:.6f}%"
    )

    return prediction_volume


# ============================================================
# 5. Contrast
# ============================================================

def get_robust_contrast(
    volume: np.ndarray,
    low_percentile: float = 0.0,
    high_percentile: float = 99.5,
) -> tuple[float, float]:

    vmin, vmax = np.percentile(
        volume,
        [
            low_percentile,
            high_percentile,
        ],
    )

    vmin = float(vmin)
    vmax = float(vmax)

    if vmax <= vmin:
        vmax = vmin + 1e-8

    return vmin, vmax


def get_contrast_limits(
    low_volume: np.ndarray,
    prediction_volume: np.ndarray,
    high_volume: np.ndarray,
    mode: str = "shared",
) -> dict[str, tuple[float, float]]:

    volumes = {
        "low": low_volume,
        "prediction": prediction_volume,
        "high": high_volume,
    }

    individual = {
        name: get_robust_contrast(
            volume,
            low_percentile=LOW_PERCENTILE,
            high_percentile=HIGH_PERCENTILE,
        )
        for name, volume in volumes.items()
    }

    if mode == "independent":
        return individual

    if mode == "shared":

        shared_vmin = min(
            limits[0]
            for limits in individual.values()
        )

        shared_vmax = max(
            limits[1]
            for limits in individual.values()
        )

        shared = (
            shared_vmin,
            shared_vmax,
        )

        return {
            "low": shared,
            "prediction": shared,
            "high": shared,
        }

    raise ValueError(
        f"Unknown contrast mode: {mode}\n"
        "Use 'independent' or 'shared'."
    )


# ============================================================
# 6. Print volume statistics
# ============================================================

def print_volume_statistics(
    name: str,
    volume: np.ndarray,
) -> None:

    print(
        f"{name:<12} | "
        f"shape={str(volume.shape):<18} | "
        f"min={volume.min():.8f} | "
        f"max={volume.max():.8f} | "
        f"mean={volume.mean():.8f} | "
        f"std={volume.std():.8f}"
    )


# ============================================================
# 7. Napari side-by-side viewer
# ============================================================

def show_napari_low_prediction_high(
    low_volume: np.ndarray,
    prediction_volume: np.ndarray,
    high_volume: np.ndarray,
    gap: int = 50,
    contrast_mode: str = "shared",
) -> None:
    """
    LOW - Prediction - HIGH를 하나의 Napari 3D 공간에
    서로 겹치지 않게 나란히 표시.

    Input volume:
        [X,Y,Z]

    Napari:
        [Z,Y,X]

    Camera:
        - fixed center
        - fixed zoom
        - no pan
        - no zoom
        - rotation remains available
    """

    expected_shape = HIGH_SHAPE

    for name, volume in (
        ("LOW", low_volume),
        ("Prediction", prediction_volume),
        ("HIGH", high_volume),
    ):
        if volume.shape != expected_shape:
            raise ValueError(
                f"\n{name} shape mismatch\n"
                f"Actual  : {volume.shape}\n"
                f"Expected: {expected_shape}"
            )

    print("\n" + "=" * 80)
    print("3D SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    print_volume_statistics(
        "LOW",
        low_volume,
    )

    print_volume_statistics(
        "Prediction",
        prediction_volume,
    )

    print_volume_statistics(
        "HIGH",
        high_volume,
    )


    # --------------------------------------------------------
    # [X,Y,Z] -> [Z,Y,X]
    # --------------------------------------------------------

    low_zyx = np.transpose(
        low_volume,
        (2, 1, 0),
    )

    prediction_zyx = np.transpose(
        prediction_volume,
        (2, 1, 0),
    )

    high_zyx = np.transpose(
        high_volume,
        (2, 1, 0),
    )

    print("\nNapari shape [Z,Y,X]")

    print(f"LOW        : {low_zyx.shape}")
    print(f"Prediction : {prediction_zyx.shape}")
    print(f"HIGH       : {high_zyx.shape}")


    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast_limits = get_contrast_limits(
        low_volume=low_volume,
        prediction_volume=prediction_volume,
        high_volume=high_volume,
        mode=contrast_mode,
    )

    print(f"\nContrast mode: {contrast_mode}")

    print(
        f"LOW        : {contrast_limits['low']}"
    )

    print(
        f"Prediction : {contrast_limits['prediction']}"
    )

    print(
        f"HIGH       : {contrast_limits['high']}"
    )


    # --------------------------------------------------------
    # Side-by-side coordinates
    # --------------------------------------------------------
    #
    # Napari:
    # [Z,Y,X]
    #
    # 마지막 X world axis 방향으로 이동.
    #
    # LOW:
    # X = 0
    #
    # Prediction:
    # X = 200 + GAP
    #
    # HIGH:
    # X = 2 * (200 + GAP)
    # --------------------------------------------------------

    x_size = low_zyx.shape[2]

    separation = x_size + gap

    low_translate = (
        0,
        0,
        0,
    )

    prediction_translate = (
        0,
        0,
        separation,
    )

    high_translate = (
        0,
        0,
        2 * separation,
    )


    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    viewer = napari.Viewer(
        ndisplay=3,
        title=(
            "E9 PAM 3D Comparison | "
            "LOW | PREDICTION | HIGH"
        ),
    )


    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    viewer.add_image(
        low_zyx,
        name="01_LOW",
        colormap=COLORMAP,
        contrast_limits=contrast_limits["low"],
        rendering=RENDERING,
        opacity=1.0,
        translate=low_translate,
        visible=True,
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    viewer.add_image(
        prediction_zyx,
        name="02_PREDICTION",
        colormap=COLORMAP,
        contrast_limits=contrast_limits["prediction"],
        rendering=RENDERING,
        opacity=1.0,
        translate=prediction_translate,
        visible=True,
    )


    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    viewer.add_image(
        high_zyx,
        name="03_HIGH",
        colormap=COLORMAP,
        contrast_limits=contrast_limits["high"],
        rendering=RENDERING,
        opacity=1.0,
        translate=high_translate,
        visible=True,
    )


    # --------------------------------------------------------
    # Initial camera angle
    # --------------------------------------------------------

    viewer.camera.angles = (
        20.0,
        -20.0,
        30.0,
    )

    viewer.camera.perspective = 0


    # --------------------------------------------------------
    # Fit all layers into view
    # --------------------------------------------------------

    viewer.reset_view(
        margin=0.05,
        reset_camera_angle=False,
    )


    # --------------------------------------------------------
    # Save fixed camera position
    # --------------------------------------------------------

    fixed_center = tuple(
        viewer.camera.center
    )

    fixed_zoom = float(
        viewer.camera.zoom
    )


    # --------------------------------------------------------
    # Disable interactive pan / zoom
    # --------------------------------------------------------

    viewer.camera.mouse_pan = False
    viewer.camera.mouse_zoom = False


    # --------------------------------------------------------
    # Force center and zoom to remain fixed.
    #
    # Only camera angles are allowed to change.
    # --------------------------------------------------------

    camera_lock_active = False

    def lock_camera_position(event=None):
        nonlocal camera_lock_active

        if camera_lock_active:
            return

        camera_lock_active = True

        try:
            if tuple(viewer.camera.center) != fixed_center:
                viewer.camera.center = fixed_center

            if not np.isclose(
                viewer.camera.zoom,
                fixed_zoom,
            ):
                viewer.camera.zoom = fixed_zoom

        finally:
            camera_lock_active = False


    viewer.camera.events.center.connect(
        lock_camera_position
    )

    viewer.camera.events.zoom.connect(
        lock_camera_position
    )


    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("\nWorld translations [Z,Y,X]")

    print(
        f"LOW        : {low_translate}"
    )

    print(
        f"Prediction : {prediction_translate}"
    )

    print(
        f"HIGH       : {high_translate}"
    )

    print("\nFixed camera")

    print(
        f"Center      : {fixed_center}"
    )

    print(
        f"Zoom        : {fixed_zoom}"
    )

    print(
        f"Angles      : {viewer.camera.angles}"
    )

    print(
        f"Mouse pan   : {viewer.camera.mouse_pan}"
    )

    print(
        f"Mouse zoom  : {viewer.camera.mouse_zoom}"
    )

    print(
        "\nCamera center and zoom are locked. "
        "3D rotation remains available."
    )

    napari.run()


# ============================================================
# 8. Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("E9 PAM 3D EVALUATION")
    print("=" * 80)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU   : {torch.cuda.get_device_name(0)}"
        )


    # --------------------------------------------------------
    # 1. Load E9 architecture
    # --------------------------------------------------------

    print("\n[1/5] Loading E9 architecture...")

    model_class = load_e9_class(
        MODEL_DEFINITION_PATH
    )


    # --------------------------------------------------------
    # 2. Load trained checkpoint
    # --------------------------------------------------------

    print("\n[2/5] Loading trained checkpoint...")

    model = load_trained_e9_model(
        model_class=model_class,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )


    # --------------------------------------------------------
    # 3. Load LOW / HIGH data
    # --------------------------------------------------------

    print("\n[3/5] Loading LOW and HIGH volumes...")

    low_adjacent_volume = load_bin_volume(
        path=LOW_PATH,
        shape=LOW_SHAPE,
        dtype=DTYPE,
        offset=OFFSET,
    )

    high_volume = load_bin_volume(
        path=HIGH_PATH,
        shape=HIGH_SHAPE,
        dtype=DTYPE,
        offset=OFFSET,
    )


    # --------------------------------------------------------
    # LOW visualization:
    #
    # [X,C,Y,Z]
    #       ↓ channel 1
    # [X,Y,Z]
    # --------------------------------------------------------

    low_center_volume = np.array(
        low_adjacent_volume[:, 1, :, :],
        dtype=np.float32,
        copy=True,
    )

    print("\nLOW center channel extracted.")

    print(
        f"3-adjacent LOW : "
        f"{low_adjacent_volume.shape}"
    )

    print(
        f"LOW center     : "
        f"{low_center_volume.shape}"
    )


    # --------------------------------------------------------
    # 4. E9 full-volume inference
    # --------------------------------------------------------

    print("\n[4/5] Predicting complete 3D volume...")

    prediction_volume = predict_full_volume(
        model=model,
        low_volume=low_adjacent_volume,
        device=device,
        batch_size=INFERENCE_BATCH_SIZE,
    )


    # --------------------------------------------------------
    # Optional:
    # Prediction volume 저장
    #
    # 필요할 경우 주석 해제.
    # --------------------------------------------------------

    # output_path = Path(
    #     "./99_model_evaluatoin/results/"
    #     "prediction_volume.npy"
    # )
    #
    # output_path.parent.mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )
    #
    # np.save(
    #     output_path,
    #     prediction_volume,
    # )
    #
    # print(
    #     f"Prediction saved: {output_path}"
    # )


    # --------------------------------------------------------
    # 5. 3D side-by-side comparison
    # --------------------------------------------------------

    print("\n[5/5] Opening Napari 3D viewer...")

    show_napari_low_prediction_high(
        low_volume=low_center_volume,
        prediction_volume=prediction_volume,
        high_volume=high_volume,
        gap=GAP,
        contrast_mode=CONTRAST_MODE,
    )


# ============================================================
# 9. Entry point
# ============================================================

if __name__ == "__main__":
    main()