import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Optional 3D viewers
import plotly.graph_objects as go
import napari


# ============================================================
# 0. Basic settings
# ============================================================
HIGH_PATH = "./../data/Train/HIGH/Train_001_HiGH.bin"
LOW_PATH  = "./../data/Train/LOW/Train_001_LOW.bin"

SHAPE = (200, 200, 512)   # [H, W, D]
DTYPE = np.uint16
OFFSET = 48


# ============================================================
# 1. Load function
# ============================================================
def load_bin_volume(path, shape=SHAPE, dtype=DTYPE, offset=OFFSET):
    raw = np.fromfile(path, dtype=dtype, offset=offset)

    expected_size = np.prod(shape)

    print(f"\nFile: {path}")
    print("Raw size:", raw.size)
    print("Expected size:", expected_size)

    if raw.size != expected_size:
        raise ValueError(
            f"Data size mismatch: raw={raw.size}, expected={expected_size}"
        )

    volume = raw.reshape(shape)

    print("Volume shape:", volume.shape)
    print("Volume min:", volume.min())
    print("Volume max:", volume.max())
    print("Volume mean:", volume.mean())

    return volume


# ============================================================
# 2. Load HIGH and LOW volumes
# ============================================================
high_volume = load_bin_volume(HIGH_PATH)
low_volume = load_bin_volume(LOW_PATH)


# ============================================================
# 3. MAP / MIP comparison
# ============================================================
def show_map_comparison(high_volume, low_volume):
    high_map = np.max(high_volume, axis=2)
    low_map = np.max(low_volume, axis=2)

    print("\nHIGH MAP shape:", high_map.shape)
    print("HIGH MAP min:", high_map.min())
    print("HIGH MAP max:", high_map.max())
    print("HIGH MAP mean:", high_map.mean())

    print("\nLOW MAP shape:", low_map.shape)
    print("LOW MAP min:", low_map.min())
    print("LOW MAP max:", low_map.max())
    print("LOW MAP mean:", low_map.mean())

    high_vmin, high_vmax = np.percentile(high_map, [0, 99])
    low_vmin, low_vmax = np.percentile(low_map, [0, 99])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(
        high_map,
        cmap="hot",
        vmin=high_vmin,
        vmax=high_vmax
    )
    axes[0].set_title("HIGH MAP / MIP")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        low_map,
        cmap="hot",
        vmin=low_vmin,
        vmax=low_vmax
    )
    axes[1].set_title("LOW MAP / MIP")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


show_map_comparison(high_volume, low_volume)


# ============================================================
# 4. Interactive depth slice viewer
# ============================================================
def show_slice_slider(volume, title="Volume"):
    vmin, vmax = np.percentile(volume, [0, 99])
    init_z = volume.shape[2] // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    img = ax.imshow(
        volume[:, :, init_z],
        cmap="hot",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"{title} | Depth slice z = {init_z}")
    ax.axis("off")
    plt.colorbar(img, ax=ax)

    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    z_slider = Slider(
        ax=slider_ax,
        label="Depth z",
        valmin=0,
        valmax=volume.shape[2] - 1,
        valinit=init_z,
        valstep=1
    )

    def update_slice(val):
        z = int(z_slider.val)
        img.set_data(volume[:, :, z])
        ax.set_title(f"{title} | Depth slice z = {z}")
        fig.canvas.draw_idle()

    z_slider.on_changed(update_slice)
    plt.show()


show_slice_slider(high_volume, title="HIGH")
show_slice_slider(low_volume, title="LOW")

# ============================================================
# 5. Napari: HIGH and LOW in one window
# ============================================================
def show_napari_high_low(high_volume, low_volume):
    """
    Napari에서 HIGH와 LOW를 하나의 창에 layer로 동시에 표시.
    """

    # Napari는 [Z, Y, X] 형태가 보기 편함
    high_zyx = np.transpose(high_volume, (2, 0, 1))
    low_zyx = np.transpose(low_volume, (2, 0, 1))

    print("\nNapari HIGH shape:", high_zyx.shape)
    print("Napari LOW shape:", low_zyx.shape)

    high_contrast = np.percentile(high_zyx, [0, 99.5])
    low_contrast = np.percentile(low_zyx, [0, 99.5])

    viewer = napari.Viewer(ndisplay=3)

    viewer.add_image(
        high_zyx,
        name="HIGH PA 3D Volume",
        colormap="hot",
        contrast_limits=high_contrast,
        rendering="mip",
        opacity=0.8,
        visible=True
    )

    viewer.add_image(
        low_zyx,
        name="LOW PA 3D Volume",
        colormap="hot",
        contrast_limits=low_contrast,
        rendering="mip",
        opacity=0.8,
        visible=True
    )

    napari.run()


show_napari_high_low(high_volume, low_volume)