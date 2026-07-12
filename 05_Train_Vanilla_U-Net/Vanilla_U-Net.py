from pathlib import Path
import argparse
import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from tqdm import tqdm


# -----------------------------
# Utility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable SSIM loss.

    pred, target:
        shape = [B, C, H, W]
        range = [0, 1]
    """
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    padding = window_size // 2

    mu_x = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=padding)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred * pred, kernel_size=window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=window_size, stride=1, padding=padding) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / (denominator + eps)
    ssim_value = ssim_map.mean().clamp(0.0, 1.0)
    loss_ssim = 1.0 - ssim_value

    return loss_ssim, ssim_value


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """
    Save tensor image.

    Input:
        tensor: [C, H, W], range [0, 1]
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    img = tensor.detach().cpu().clamp(0, 1)

    if img.ndim != 3:
        raise ValueError(f"Expected tensor shape [C, H, W], got {img.shape}")

    TF.to_pil_image(img).save(path)


def save_hot_image(img: np.ndarray, path: Path, title: str = "") -> None:
    """
    Save numpy image with hot colormap for visual checking.
    This is only for visualization, not for preserving raw intensity values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="hot", vmin=0.0, vmax=1.0)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_model_dirs(models_dir: Path):
    checkpoint_dir = models_dir / "checkpoints"
    best_dir = models_dir / "best"
    latest_dir = models_dir / "latest"
    final_dir = models_dir / "final"

    for d in [checkpoint_dir, best_dir, latest_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    paths = {
        "checkpoint_dir": checkpoint_dir,
        "best_model": best_dir / "model_best.pth",
        "latest_model": latest_dir / "model_latest.pth",
        "latest_checkpoint": latest_dir / "checkpoint_latest.pth",
        "final_model": final_dir / "model_final.pth",
    }
    return paths


# -----------------------------
# Bin loading functions
# -----------------------------

def read_bin_volume(
    path: Path,
    shape: Tuple[int, int, int] = (200, 200, 512),
    dtype=np.uint16,
    offset: int = 48,
) -> np.ndarray:
    """
    Read one raw .bin volume.

    Expected stored order:
        [H, W, D]

    Return:
        volume: float32 numpy array, shape [H, W, D]
    """
    expected_elements = int(np.prod(shape))
    expected_size = expected_elements * np.dtype(dtype).itemsize + offset

    file_size = path.stat().st_size
    if file_size != expected_size:
        raise ValueError(
            f"File size mismatch: {path}\n"
            f"Expected: {expected_size} bytes\n"
            f"Actual  : {file_size} bytes\n"
            f"Check height, width, depth, dtype, or offset."
        )

    raw = np.fromfile(path, dtype=dtype, offset=offset)

    if raw.size != expected_elements:
        raise ValueError(
            f"Element count mismatch: {path}\n"
            f"Expected: {expected_elements}\n"
            f"Actual  : {raw.size}"
        )

    volume = raw.reshape(shape)  # [H, W, D]
    return volume.astype(np.float32)


def make_projection(volume: np.ndarray, projection: str = "p99") -> np.ndarray:
    """
    Convert [H, W, D] volume to [H, W] MAP image.

    projection:
        max  : maximum intensity projection
        p99  : 99th percentile projection, more robust to spike noise
        mean : mean projection
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape [H, W, D], got {volume.shape}")

    if projection == "max":
        img = np.max(volume, axis=2)
    elif projection == "p99":
        img = np.percentile(volume, 99, axis=2)
    elif projection == "mean":
        img = np.mean(volume, axis=2)
    else:
        raise ValueError(f"Unsupported projection: {projection}")

    return img.astype(np.float32)


def normalize_image(img: np.ndarray, method: str = "percentile") -> np.ndarray:
    """
    Normalize 2D image to [0, 1].

    method:
        minmax     : use actual min/max
        percentile : clip with 1st and 99th percentiles before min-max scaling
    """
    img = img.astype(np.float32)

    if method == "minmax":
        v_min = float(img.min())
        v_max = float(img.max())

    elif method == "percentile":
        v_min, v_max = np.percentile(img, [1, 99])
        v_min = float(v_min)
        v_max = float(v_max)
        img = np.clip(img, v_min, v_max)

    else:
        raise ValueError(f"Unsupported normalize method: {method}")

    if v_max > v_min:
        img = (img - v_min) / (v_max - v_min)
    else:
        img = img * 0.0

    return img.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------

class PairedBinMAPDataset(Dataset):
    """
    LOW .bin volume  -> input MAP image
    HIGH .bin volume -> target MAP image

    Output:
        low  : [1, H, W], float32, [0, 1]
        high : [1, H, W], float32, [0, 1]
    """

    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        shape: Tuple[int, int, int] = (200, 200, 512),
        dtype=np.uint16,
        offset: int = 48,
        projection: str = "p99",
        normalize: str = "percentile",
        augment: bool = True,
    ):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
        self.projection = projection
        self.normalize = normalize
        self.augment = augment

        low_files = sorted(self.low_dir.glob("*.bin"))
        high_files = sorted(self.high_dir.glob("*.bin"))

        self.pairs = self._make_pairs(low_files, high_files)

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired .bin files found.\n"
                f"LOW dir : {self.low_dir}\n"
                f"HIGH dir: {self.high_dir}\n"
                f"Expected examples:\n"
                f"  Train_000_LOW.bin\n"
                f"  Train_000_HIGH.bin"
            )

        print(f"Found {len(self.pairs)} LOW/HIGH bin pairs.")

    @staticmethod
    def _normalize_stem(path: Path) -> str:
        """
        Train_000_LOW.bin  -> Train_000
        Train_000_HIGH.bin -> Train_000
        """
        stem = path.stem
        stem = stem.replace("_LOW", "")
        stem = stem.replace("_HIGH", "")
        stem = stem.replace("_low", "")
        stem = stem.replace("_high", "")
        return stem

    def _make_pairs(self, low_files: List[Path], high_files: List[Path]) -> List[Tuple[Path, Path]]:
        high_map = {self._normalize_stem(p): p for p in high_files}

        pairs = []
        for low_path in low_files:
            key = self._normalize_stem(low_path)
            high_path = high_map.get(key)

            if high_path is not None:
                pairs.append((low_path, high_path))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        low_path, high_path = self.pairs[idx]

        low_volume = read_bin_volume(
            low_path,
            shape=self.shape,
            dtype=self.dtype,
            offset=self.offset,
        )
        high_volume = read_bin_volume(
            high_path,
            shape=self.shape,
            dtype=self.dtype,
            offset=self.offset,
        )

        low_map = make_projection(low_volume, projection=self.projection)
        high_map = make_projection(high_volume, projection=self.projection)

        low_map = normalize_image(low_map, method=self.normalize)
        high_map = normalize_image(high_map, method=self.normalize)

        low = torch.from_numpy(low_map).unsqueeze(0)    # [1, H, W]
        high = torch.from_numpy(high_map).unsqueeze(0)  # [1, H, W]

        if self.augment:
            if random.random() < 0.5:
                low = torch.flip(low, dims=[2])
                high = torch.flip(high, dims=[2])

            if random.random() < 0.5:
                low = torch.flip(low, dims=[1])
                high = torch.flip(high, dims=[1])

        return low, high


# -----------------------------
# Model
# -----------------------------

class UNetConv2(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEnhancer(nn.Module):
    """
    2D U-Net for grayscale MAP enhancement.

    Input:
        [B, 1, H, W]

    Output:
        [B, 1, H, W]
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()

        b = base_channels

        self.conv_1 = UNetConv2(in_channels, b)
        self.conv_2 = UNetConv2(b, b * 2)
        self.conv_3 = UNetConv2(b * 2, b * 4)
        self.conv_4 = UNetConv2(b * 4, b * 8)

        self.mid_conv = UNetConv2(b * 8, b * 16)

        self.up_1 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.up_2 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.up_3 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.up_4 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)

        self.conv_5 = UNetConv2(b * 16, b * 8)
        self.conv_6 = UNetConv2(b * 8, b * 4)
        self.conv_7 = UNetConv2(b * 4, b * 2)
        self.conv_8 = UNetConv2(b * 2, b)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.end = nn.Conv2d(b, out_channels, kernel_size=1, stride=1)

    @staticmethod
    def center_crop_like(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, h, w = src.shape
        _, _, th, tw = target.shape

        top = max((h - th) // 2, 0)
        left = max((w - tw) // 2, 0)

        return src[:, :, top: top + th, left: left + tw]

    @staticmethod
    def pad_to_even(x: torch.Tensor) -> torch.Tensor:
        pad_h = x.size(2) % 2
        pad_w = x.size(3) % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Padding compensates for valid convolutions in the original U-Net style.
        padded_x = F.pad(x, (92, 92, 92, 92), mode="reflect")

        conv_1 = self.conv_1(padded_x)
        pool1 = self.down(self.pad_to_even(conv_1))

        conv_2 = self.conv_2(pool1)
        pool2 = self.down(self.pad_to_even(conv_2))

        conv_3 = self.conv_3(pool2)
        pool3 = self.down(self.pad_to_even(conv_3))

        conv_4 = self.conv_4(pool3)
        pool4 = self.down(self.pad_to_even(conv_4))

        mid = self.mid_conv(pool4)

        up_1 = self.up_1(mid)
        up_1 = torch.cat([up_1, self.center_crop_like(conv_4, up_1)], dim=1)
        conv_5 = self.conv_5(up_1)

        up_2 = self.up_2(conv_5)
        up_2 = torch.cat([up_2, self.center_crop_like(conv_3, up_2)], dim=1)
        conv_6 = self.conv_6(up_2)

        up_3 = self.up_3(conv_6)
        up_3 = torch.cat([up_3, self.center_crop_like(conv_2, up_3)], dim=1)
        conv_7 = self.conv_7(up_3)

        up_4 = self.up_4(conv_7)
        up_4 = torch.cat([up_4, self.center_crop_like(conv_1, up_4)], dim=1)
        conv_8 = self.conv_8(up_4)

        out = self.end(conv_8)
        out = self.center_crop_like(out, x)

        return torch.sigmoid(out)


# -----------------------------
# Checkpoint
# -----------------------------

def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best_val_loss: float) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model, optimizer, scaler, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))

    return start_epoch, best_val_loss


# -----------------------------
# Preview
# -----------------------------

def preview_pair(args) -> None:
    """
    Save one LOW/HIGH MAP pair for debugging.
    """
    low_files = sorted(Path(args.low_dir).glob("*.bin"))
    high_files = sorted(Path(args.high_dir).glob("*.bin"))

    if len(low_files) == 0:
        raise RuntimeError(f"No LOW .bin files found: {args.low_dir}")
    if len(high_files) == 0:
        raise RuntimeError(f"No HIGH .bin files found: {args.high_dir}")

    dataset = PairedBinMAPDataset(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        shape=(args.height, args.width, args.depth),
        dtype=np.uint16,
        offset=args.offset,
        projection=args.projection,
        normalize=args.normalize,
        augment=False,
    )

    low, high = dataset[args.preview_index]

    low_np = low.squeeze(0).numpy()
    high_np = high.squeeze(0).numpy()

    preview_dir = Path(args.preview_dir)
    save_hot_image(low_np, preview_dir / "low_map_hot.png", title="LOW MAP")
    save_hot_image(high_np, preview_dir / "high_map_hot.png", title="HIGH MAP")

    print(f"Saved preview images to: {preview_dir}")


# -----------------------------
# Train / Validate
# -----------------------------

def train(args) -> None:
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    models_dir = Path(args.models_dir)
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    model_paths = make_model_dirs(models_dir)
    checkpoint_dir = model_paths["checkpoint_dir"]
    best_model_path = model_paths["best_model"]
    latest_model_path = model_paths["latest_model"]
    latest_checkpoint_path = model_paths["latest_checkpoint"]
    final_model_path = model_paths["final_model"]

    train_base_dataset = PairedBinMAPDataset(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        shape=(args.height, args.width, args.depth),
        dtype=np.uint16,
        offset=args.offset,
        projection=args.projection,
        normalize=args.normalize,
        augment=True,
    )

    val_base_dataset = PairedBinMAPDataset(
        low_dir=args.low_dir,
        high_dir=args.high_dir,
        shape=(args.height, args.width, args.depth),
        dtype=np.uint16,
        offset=args.offset,
        projection=args.projection,
        normalize=args.normalize,
        augment=False,
    )

    total_len = len(train_base_dataset)
    train_len = int(total_len * args.train_ratio)
    val_len = total_len - train_len

    if train_len <= 0 or val_len <= 0:
        raise ValueError(
            f"Invalid split. total={total_len}, train={train_len}, val={val_len}. "
            f"Adjust --train_ratio or add more data."
        )

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total_len, generator=generator).tolist()

    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    train_dataset = Subset(train_base_dataset, train_indices)
    val_dataset = Subset(val_base_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = UNetEnhancer(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume_path) if args.resume_path else latest_checkpoint_path

        if resume_path.exists():
            start_epoch, best_val_loss = load_checkpoint(
                resume_path,
                model,
                optimizer,
                scaler,
                device,
            )
            print(f"Loaded checkpoint: {resume_path}")
            print(f"Resume from epoch {start_epoch + 1}")
        else:
            print(f"Resume requested, but checkpoint not found: {resume_path}")
            print("Start new training.")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for phase in ["train", "val"]:
            is_train = phase == "train"

            if is_train:
                model.train()
            else:
                model.eval()

            loader = train_loader if is_train else val_loader

            running_loss = 0.0
            running_psnr = 0.0
            running_ssim = 0.0
            n_samples = 0

            pbar = tqdm(loader, desc=phase.capitalize(), total=len(loader))

            for batch_idx, (low, high) in enumerate(pbar):
                low = low.to(device, non_blocking=True)
                high = high.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                        pred = model(low)

                        loss_l1 = l1_loss(pred, high)
                        loss_mse = mse_loss(pred, high)
                        loss_ssim, ssim_value = ssim_loss(
                            pred,
                            high,
                            window_size=args.ssim_window_size,
                            data_range=1.0,
                        )

                        loss = (
                            args.l1_weight * loss_l1
                            + args.mse_weight * loss_mse
                            + args.ssim_weight * loss_ssim
                        )

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                batch_size = low.size(0)
                batch_psnr = psnr(pred.detach(), high.detach()).item()

                running_loss += loss.item() * batch_size
                running_psnr += batch_psnr * batch_size
                running_ssim += ssim_value.detach().item() * batch_size
                n_samples += batch_size

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "L1": f"{loss_l1.item():.4f}",
                    "MSE": f"{loss_mse.item():.4f}",
                    "SSIM": f"{ssim_value.detach().item():.4f}",
                    "PSNR": f"{batch_psnr:.2f}",
                })

                if (
                    not is_train
                    and batch_idx == 0
                    and ((epoch + 1) % args.save_sample_every == 0)
                ):
                    save_tensor_image(low[0], sample_dir / f"epoch_{epoch + 1:03d}_input.png")
                    save_tensor_image(pred[0], sample_dir / f"epoch_{epoch + 1:03d}_pred.png")
                    save_tensor_image(high[0], sample_dir / f"epoch_{epoch + 1:03d}_target.png")

            epoch_loss = running_loss / max(n_samples, 1)
            epoch_psnr = running_psnr / max(n_samples, 1)
            epoch_ssim = running_ssim / max(n_samples, 1)

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.6f} | "
                f"PSNR: {epoch_psnr:.2f} dB | "
                f"SSIM: {epoch_ssim:.4f}"
            )

            if phase == "val" and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model: {best_model_path}")

        epoch_checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"

        save_checkpoint(
            epoch_checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            best_val_loss,
        )

        save_checkpoint(
            latest_checkpoint_path,
            model,
            optimizer,
            scaler,
            epoch,
            best_val_loss,
        )

        torch.save(model.state_dict(), latest_model_path)

        print(f"Saved epoch checkpoint : {epoch_checkpoint_path}")
        print(f"Saved latest checkpoint: {latest_checkpoint_path}")
        print(f"Saved latest model     : {latest_model_path}")

    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining finished. Saved final model: {final_model_path}")


# -----------------------------
# Inference for one bin
# -----------------------------

@torch.no_grad()
def infer_bin(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = UNetEnhancer(
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    input_path = Path(args.input)
    output_path = Path(args.output)

    volume = read_bin_volume(
        input_path,
        shape=(args.height, args.width, args.depth),
        dtype=np.uint16,
        offset=args.offset,
    )

    map_img = make_projection(volume, projection=args.projection)
    map_img = normalize_image(map_img, method=args.normalize)

    x = torch.from_numpy(map_img).float().unsqueeze(0).unsqueeze(0).to(device)

    pred = model(x)[0]

    save_tensor_image(pred, output_path)

    if args.save_hot_output:
        pred_np = pred.squeeze(0).detach().cpu().clamp(0, 1).numpy()
        hot_path = output_path.with_name(output_path.stem + "_hot.png")
        save_hot_image(pred_np, hot_path, title="Enhanced MAP")

    print(f"Saved enhanced MAP image: {output_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="U-Net MAP Enhancement Training for Paired .bin Volume Data"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer_bin", "preview"],
    )

    # Data paths
    parser.add_argument("--low_dir", type=str, default="./data/Train/LOW")
    parser.add_argument("--high_dir", type=str, default="./data/Train/HIGH")

    # Bin volume parameters
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--offset", type=int, default=48)
    parser.add_argument(
        "--projection",
        type=str,
        default="p99",
        choices=["max", "p99", "mean"],
        help="Projection method from [H, W, D] volume to 2D MAP image.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="percentile",
        choices=["minmax", "percentile"],
        help="Normalization method for MAP image.",
    )

    # Save paths
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--sample_dir", type=str, default="./outputs/bin_map_samples")
    parser.add_argument("--preview_dir", type=str, default="./outputs/bin_map_preview")
    parser.add_argument("--preview_index", type=int, default=0)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume_path", type=str, default="", help="Optional checkpoint path")

    # Loss weights
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.1)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--ssim_window_size", type=int, default=11)
    parser.add_argument("--save_sample_every", type=int, default=5)

    # Inference parameters
    parser.add_argument("--weights", type=str, default="./models_bin_map_enhancement/best/model_best.pth")
    parser.add_argument("--input", type=str, default="./data/Train/LOW/Train_000_LOW.bin")
    parser.add_argument("--output", type=str, default="./outputs/Train_000_enhanced_map.png")
    parser.add_argument("--save_hot_output", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "infer_bin":
        infer_bin(args)

    elif args.mode == "preview":
        preview_pair(args)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()