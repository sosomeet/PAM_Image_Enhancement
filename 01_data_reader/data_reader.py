import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import napari


# ============================================================
# 0. Basic settings
# ============================================================
HIGH_PATH = "./data/Train/HIGH/Train_001_HiGH.bin"
LOW_PATH = "./data/Train/LOW/Train_001_LOW.bin"

SHAPE = (200, 200, 512)   # (X, Y, Z)
DTYPE = np.uint16
OFFSET = 48


# ============================================================
# 1. Load function
# ============================================================
def load_bin_volume(
    path,
    shape=SHAPE,
    dtype=DTYPE,
    offset=OFFSET
):
    """
    .bin 파일에서 48-byte header를 제외한 뒤
    3D volume 데이터를 읽습니다.
    """
    raw = np.fromfile(
        path,
        dtype=dtype,
        offset=offset
    )

    expected_size = int(np.prod(shape))

    print(f"\nFile: {path}")
    print("Raw size:", raw.size)
    print("Expected size:", expected_size)

    if raw.size != expected_size:
        raise ValueError(
            f"Data size mismatch: "
            f"raw={raw.size}, expected={expected_size}"
        )

    volume = raw.reshape(shape)

    print("Volume shape:", volume.shape)
    print("Volume min:", volume.min())
    print("Volume max:", volume.max())
    print("Volume mean:", volume.mean())

    return volume


# ============================================================
# 2. Shared colormap limits
# ============================================================
def get_shared_limits(
    high_data,
    low_data,
    lower_percentile=0,
    upper_percentile=99
):
    """
    HIGH와 LOW 데이터에 공통으로 적용할 표시 범위를 계산합니다.

    두 데이터를 따로 percentile 계산한 뒤,
    더 작은 하한값과 더 큰 상한값을 공통 범위로 사용합니다.
    """
    high_limits = np.percentile(
        high_data,
        [lower_percentile, upper_percentile]
    )

    low_limits = np.percentile(
        low_data,
        [lower_percentile, upper_percentile]
    )

    shared_vmin = min(high_limits[0], low_limits[0])
    shared_vmax = max(high_limits[1], low_limits[1])

    return float(shared_vmin), float(shared_vmax)


# ============================================================
# 3. X-Y plane MAP / MIP comparison
# ============================================================
def show_xy_map_comparison(high_volume, low_volume):
    """
    Z축 방향으로 Maximum Intensity Projection을 수행하여
    X-Y plane MAP을 생성합니다.

    Input shape : (X, Y, Z)
    Output shape: (X, Y)
    """
    high_xy_map = np.max(high_volume, axis=2)
    low_xy_map = np.max(low_volume, axis=2)

    shared_vmin, shared_vmax = get_shared_limits(
        high_xy_map,
        low_xy_map,
        lower_percentile=0,
        upper_percentile=99
    )

    print("\nHIGH X-Y MAP shape:", high_xy_map.shape)
    print("HIGH X-Y MAP min:", high_xy_map.min())
    print("HIGH X-Y MAP max:", high_xy_map.max())
    print("HIGH X-Y MAP mean:", high_xy_map.mean())

    print("\nLOW X-Y MAP shape:", low_xy_map.shape)
    print("LOW X-Y MAP min:", low_xy_map.min())
    print("LOW X-Y MAP max:", low_xy_map.max())
    print("LOW X-Y MAP mean:", low_xy_map.mean())

    print("\nX-Y shared vmin:", shared_vmin)
    print("X-Y shared vmax:", shared_vmax)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True
    )

    im0 = axes[0].imshow(
        high_xy_map,
        cmap="hot",
        vmin=shared_vmin,
        vmax=shared_vmax
    )
    axes[0].set_title("HIGH X-Y MAP / MIP")
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("X")

    axes[1].imshow(
        low_xy_map,
        cmap="hot",
        vmin=shared_vmin,
        vmax=shared_vmax
    )
    axes[1].set_title("LOW X-Y MAP / MIP")
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("X")

    # 두 영상이 같은 범위를 사용하므로 colorbar도 하나만 표시
    fig.colorbar(
        im0,
        ax=axes,
        fraction=0.03,
        pad=0.04,
        label="Intensity"
    )

    plt.show()


# ============================================================
# 4. Interactive depth slice viewer
# ============================================================
def show_slice_slider(
    volume,
    title,
    shared_vmin,
    shared_vmax
):
    """
    특정 Z 위치의 X-Y slice를 slider로 확인합니다.

    HIGH와 LOW viewer에 동일한 vmin, vmax를 전달하여
    같은 intensity가 같은 색으로 표시되도록 합니다.
    """
    init_z = volume.shape[2] // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    img = ax.imshow(
        volume[:, :, init_z],
        cmap="hot",
        vmin=shared_vmin,
        vmax=shared_vmax
    )

    ax.set_title(f"{title} | Depth slice z = {init_z}")
    ax.axis("off")

    plt.colorbar(
        img,
        ax=ax,
        label="Intensity"
    )

    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])

    z_slider = Slider(
        ax=slider_ax,
        label="Depth z",
        valmin=0,
        valmax=volume.shape[2] - 1,
        valinit=init_z,
        valstep=1
    )

    def update_slice(_):
        z = int(z_slider.val)

        img.set_data(volume[:, :, z])
        ax.set_title(f"{title} | Depth slice z = {z}")

        fig.canvas.draw_idle()

    z_slider.on_changed(update_slice)

    plt.show()


# ============================================================
# 5. Rotated Y-Z plane MAP / MIP comparison
# ============================================================
def show_yz_map_comparison(high_volume, low_volume):
    """
    X축 방향으로 Maximum Intensity Projection을 수행하여
    Y-Z plane MAP을 생성합니다.

    생성한 Y-Z MAP은 오른쪽으로 90도 회전합니다.

    Input shape        : (X, Y, Z)
    Original MAP shape : (Y, Z)
    Rotated MAP shape  : (Z, Y)
    """
    high_yz_map = np.max(high_volume, axis=0)
    low_yz_map = np.max(low_volume, axis=0)

    # 오른쪽, 즉 시계 방향으로 90도 회전
    high_yz_map = np.rot90(high_yz_map, k=-1)
    low_yz_map = np.rot90(low_yz_map, k=-1)

    shared_vmin, shared_vmax = get_shared_limits(
        high_yz_map,
        low_yz_map,
        lower_percentile=0,
        upper_percentile=99
    )

    print("\nHIGH rotated Y-Z MAP shape:", high_yz_map.shape)
    print("HIGH Y-Z MAP min:", high_yz_map.min())
    print("HIGH Y-Z MAP max:", high_yz_map.max())
    print("HIGH Y-Z MAP mean:", high_yz_map.mean())

    print("\nLOW rotated Y-Z MAP shape:", low_yz_map.shape)
    print("LOW Y-Z MAP min:", low_yz_map.min())
    print("LOW Y-Z MAP max:", low_yz_map.max())
    print("LOW Y-Z MAP mean:", low_yz_map.mean())

    print("\nY-Z shared vmin:", shared_vmin)
    print("Y-Z shared vmax:", shared_vmax)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 8),
        constrained_layout=True
    )

    im0 = axes[0].imshow(
        high_yz_map,
        cmap="hot",
        vmin=shared_vmin,
        vmax=shared_vmax,
        aspect="auto"
    )
    axes[0].set_title("HIGH Y-Z MAP / MIP")
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("Z")

    axes[1].imshow(
        low_yz_map,
        cmap="hot",
        vmin=shared_vmin,
        vmax=shared_vmax,
        aspect="auto"
    )
    axes[1].set_title("LOW Y-Z MAP / MIP")
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("Z")

    fig.colorbar(
        im0,
        ax=axes,
        fraction=0.03,
        pad=0.04,
        label="Intensity"
    )

    plt.show()


# ============================================================
# 6. Napari: HIGH and LOW in one window
# ============================================================
def show_napari_high_low(high_volume, low_volume):
    """
    Napari에서 HIGH와 LOW를 하나의 창에 layer로 표시합니다.

    두 layer에 동일한 contrast_limits를 적용합니다.
    """
    # Napari에서는 (Z, Y, X) 순서가 3D 확인에 편리함
    high_zyx = np.transpose(high_volume, (2, 1, 0))
    low_zyx = np.transpose(low_volume, (2, 1, 0))

    shared_vmin, shared_vmax = get_shared_limits(
        high_zyx,
        low_zyx,
        lower_percentile=0,
        upper_percentile=99.5
    )

    shared_contrast = (
        shared_vmin,
        shared_vmax
    )

    print("\nNapari HIGH shape:", high_zyx.shape)
    print("Napari LOW shape:", low_zyx.shape)
    print("Napari shared contrast:", shared_contrast)

    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(
        high_zyx,
        name="HIGH PA 3D Volume",
        colormap="hot",
        contrast_limits=shared_contrast,
        rendering="mip",
        opacity=0.8,
        visible=True
    )

    viewer.add_image(
        low_zyx,
        name="LOW PA 3D Volume",
        colormap="hot",
        contrast_limits=shared_contrast,
        rendering="mip",
        opacity=0.8,
        visible=False
    )

    napari.run()


# ============================================================
# 7. Main
# ============================================================
def main():
    # HIGH 및 LOW volume 읽기
    high_volume = load_bin_volume(HIGH_PATH)
    low_volume = load_bin_volume(LOW_PATH)

    # X-Y plane MAP
    show_xy_map_comparison(
        high_volume,
        low_volume
    )

    # Slice viewer에 적용할 공통 colormap 범위
    slice_vmin, slice_vmax = get_shared_limits(
        high_volume,
        low_volume,
        lower_percentile=0,
        upper_percentile=99
    )

    print("\nSlice shared vmin:", slice_vmin)
    print("Slice shared vmax:", slice_vmax)

    # HIGH와 LOW slice viewer
    show_slice_slider(
        high_volume,
        title="HIGH",
        shared_vmin=slice_vmin,
        shared_vmax=slice_vmax
    )

    show_slice_slider(
        low_volume,
        title="LOW",
        shared_vmin=slice_vmin,
        shared_vmax=slice_vmax
    )

    # 오른쪽으로 90도 회전한 Y-Z plane MAP
    show_yz_map_comparison(
        high_volume,
        low_volume
    )

    # 마지막으로 Napari 3D volume 확인
    show_napari_high_low(
        high_volume,
        low_volume
    )


if __name__ == "__main__":
    main()