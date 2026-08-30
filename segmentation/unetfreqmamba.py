import torch
torch.cuda.empty_cache()
import os
import random
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from datetime import datetime
from medpy.metric.binary import hd95, asd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.cuda import amp

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# ============ Configuration ============
CONFIG = {
    # Set these before running, or override them from the command line:
    #   SEG_IMAGE_DIR / SEG_MASK_DIR environment variables, or
    #   --image_dir / --mask_dir
    "image_dir": os.environ.get("SEG_IMAGE_DIR", "data/images"),
    "mask_dir":  os.environ.get("SEG_MASK_DIR",  "data/masks"),
    "image_size": (512, 688),
    "batch_size": 2,
    "num_epochs": 150,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "subset_size": None,
    "n_folds": 5,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "patience": 20,
    "use_tensorboard": True,
    "output_dir": "./outputs_FreqMamba",   # one directory per variant
    "test_ratio": 0.15,
}

# ============ Random seeds ============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

# ============ Dataset ============
class ShellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512,688), augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir)
                                  if f.lower().endswith(('.png','.tif','.tiff'))])

        self.pairs = []
        for img_f in self.image_files:
            stem = os.path.splitext(img_f)[0]
            for ext in ['.png','.tif','.tiff']:
                mask_f = stem + ext
                if mask_f in self.mask_files:
                    self.pairs.append((img_f, mask_f))
                    break
        if not self.pairs:
            raise RuntimeError("No image-mask pairs found.")

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, mask_name))
        if mask.mode == 'RGBA':
            mask = mask.split()[3]
        else:
            mask = mask.convert('L')

        target_h, target_w = self.image_size
        image = image.resize((target_w, target_h), Image.BILINEAR)
        mask = mask.resize((target_w, target_h), Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Image.BILINEAR)
            mask = mask.rotate(angle, resample=Image.NEAREST)
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.95, 1.05))

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask, img_name

def create_subset(dataset, subset_size, seed=42):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    return Subset(dataset, indices[:subset_size])

# ============ Frequency enhancement module (FFE) ============
class FrequencyEnhancementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels * 2, channels * 2, 1)
        self.freq_norm = nn.InstanceNorm2d(channels * 2, affine=True)
        self.bn = nn.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.filter = nn.Parameter(torch.randn(1, channels, 1, 1, 2) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        real, imag = x_fft.real, x_fft.imag
        x_freq = torch.cat([real, imag], dim=1)
        x_freq = self.conv1x1(x_freq)
        x_freq = self.freq_norm(x_freq)
        x_freq = self.bn(x_freq)
        x_freq = self.relu(x_freq)
        real_out, imag_out = torch.chunk(x_freq, 2, dim=1)
        x_freq_complex = torch.complex(real_out, imag_out)
        filter_complex = torch.complex(self.filter[...,0], self.filter[...,1])
        x_freq_complex = x_freq_complex * filter_complex
        x_out = torch.fft.irfft2(x_freq_complex, s=(H, W), norm='ortho')
        return x_out

# ============ Mamba2 ============
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("error: mamba_ssm (which provides Mamba2) is required")

class SS2D_Radial(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, shared=False):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm is not installed")
        self.mamba = Mamba2(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.register_buffer('sorted_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('inverse_indices', torch.tensor([], dtype=torch.long))
        self.cached_H, self.cached_W = 0, 0

    def _get_indices(self, H, W, device):
        if H == self.cached_H and W == self.cached_W and self.sorted_indices.numel() > 0:
            return self.sorted_indices, self.inverse_indices
        cy, cx = H // 2, W // 2
        y, x_coord = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dist = (y - cy) ** 2 + (x_coord - cx) ** 2
        sorted_idx = np.argsort(dist.ravel())
        inverse_idx = np.argsort(sorted_idx)
        sorted_idx_t = torch.tensor(sorted_idx, dtype=torch.long, device=device)
        inverse_idx_t = torch.tensor(inverse_idx, dtype=torch.long, device=device)
        self.cached_H, self.cached_W = H, W
        self.sorted_indices = sorted_idx_t
        self.inverse_indices = inverse_idx_t
        return sorted_idx_t, inverse_idx_t

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        sorted_idx, inverse_idx = self._get_indices(H, W, device)
        x_flat = x.flatten(2)
        x_sorted = x_flat[:, :, sorted_idx]
        seq = x_sorted.transpose(1, 2)
        seq = self.mamba(seq)
        out_flat = seq.transpose(1, 2)
        out_restored = out_flat[:, :, inverse_idx]
        out = out_restored.reshape(B, C, H, W)
        return x + out

class VisualStateSpaceBlock(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, shared=False):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.ss2d = SS2D_Radial(dim, d_state=d_state, d_conv=d_conv, expand=expand, shared=shared)
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, x):
        if x.is_contiguous():
            x = x.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x1 = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x1 = self.linear1(x1)
        x1 = self.silu(x1)
        x2 = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x2 = self.linear2(x2)
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
        x2 = self.dwconv(x2)
        x2 = self.silu(x2)
        x2 = self.ss2d(x2)
        x2 = x2.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x2 = self.norm(x2)
        out = x1 * x2
        out = self.out_linear(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out

class FreqEMBlock(nn.Module):
    def __init__(self, channels, d_state=64, d_conv=4, expand=2, shared=False):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.vss = VisualStateSpaceBlock(channels // 2, d_state=d_state, d_conv=d_conv, expand=expand, shared=shared)
        self.ffe = FrequencyEnhancementModule(channels // 2)
        self.gate_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.center_predictor = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        self.sigma_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        B, C, H, W = x.shape

        weight_map = self.center_predictor(x)
        weight_sum = weight_map.sum(dim=(2,3), keepdim=True) + 1e-6
        y_coord = torch.linspace(0, 1, H, device=x.device).view(1,1,H,1)
        x_coord = torch.linspace(0, 1, W, device=x.device).view(1,1,1,W)
        center_y = (weight_map * y_coord).sum(dim=(2,3), keepdim=True) / weight_sum
        center_x = (weight_map * x_coord).sum(dim=(2,3), keepdim=True) / weight_sum
        sigma = self.sigma_predictor(x).view(B,1,1,1).clamp(0.05, 0.5)
        dist = (y_coord - center_y)**2 + (x_coord - center_x)**2
        dynamic_mask = torch.exp(-dist / (2 * sigma**2))
        x = x * dynamic_mask

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = x1.permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        x1 = self.vss(x1)
        x2 = self.ffe(x2)

        x_cat = torch.cat([x1, x2], dim=1)
        x_cat = identity + x_cat

        gate = torch.sigmoid(self.gate_conv(x_cat))
        x_cat = x_cat * (1 + gate)

        x_cat = x_cat.permute(0, 2, 3, 1)
        x_cat = self.norm(x_cat)
        x_cat = x_cat.permute(0, 3, 1, 2)
        x_cat = self.dwconv(x_cat)
        x_cat = self.bn(x_cat)
        x_cat = self.relu(x_cat)

        return x_cat + identity

# ============ U-Net building blocks ============
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.mp(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet_FreqMamba(nn.Module):
    def __init__(self, base=64, shared=False):
        super().__init__()
        self.inc = DoubleConv(3, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.freqem3 = FreqEMBlock(base*8, shared=shared)
        self.down4 = Down(base*8, base*16)
        self.freqem4 = FreqEMBlock(base*16, shared=shared)
        self.up1 = Up(base*16 + base*8, base*8)
        self.up2 = Up(base*8 + base*4, base*4)
        self.up3 = Up(base*4 + base*2, base*2)
        self.up4 = Up(base*2 + base, base)
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.freqem3(x4)
        x5 = self.down4(x4)
        x5 = self.freqem4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# ============ Loss function ============
class DiceBCELoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice = 1 - (2*inter + self.eps) / (pred.sum() + target.sum() + self.eps)
        return self.alpha * bce + (1 - self.alpha) * dice

# ============ Metrics ============
def boundary_f1(pred, target, threshold=3):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, H, W = pred.shape
    pred = pred.float()
    target = target.float()
    kernel_size = 2 * threshold + 1
    padding = threshold
    pred_pad = F.pad(pred, (padding, padding, padding, padding), mode='constant', value=0)
    target_pad = F.pad(target, (padding, padding, padding, padding), mode='constant', value=0)
    pred_dilated = F.max_pool2d(pred_pad, kernel_size, stride=1, padding=0)
    target_dilated = F.max_pool2d(target_pad, kernel_size, stride=1, padding=0)
    pred_boundary = pred_dilated - pred
    target_boundary = target_dilated - target
    pred_boundary_bin = (pred_boundary > 0.5).float()
    target_boundary_bin = (target_boundary > 0.5).float()
    tp = (pred_boundary_bin * target_boundary_bin).sum(dim=(1, 2))
    fp = (pred_boundary_bin * (1 - target_boundary_bin)).sum(dim=(1, 2))
    fn = ((1 - pred_boundary_bin) * target_boundary_bin).sum(dim=(1, 2))
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1.mean().item()

def calc_metrics(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = target.float()
    batch_size = pred.shape[0]
    
    iou_sum, dice_sum = 0.0, 0.0
    for b in range(batch_size):
        p = pred[b, 0]; t = target[b, 0]
        inter = (p * t).sum()
        union = (p + t - p * t).sum() + 1e-7
        iou_sum += (inter / union).item()
        dice_sum += (2 * inter / (p.sum() + t.sum() + 1e-7)).item()
    iou = iou_sum / batch_size
    dice = dice_sum / batch_size

    pred_np = pred.cpu().numpy().flatten().astype(int)
    target_np = target.cpu().numpy().flatten().astype(int)
    tn, fp, fn, tp = confusion_matrix(target_np, pred_np, labels=[0,1]).ravel()
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    hd95_list, asd_list = [], []
    for b in range(batch_size):
        p = pred[b, 0].cpu().numpy().astype(np.bool_)
        t = target[b, 0].cpu().numpy().astype(np.bool_)
        if p.sum() == 0 and t.sum() == 0:
            hd95_list.append(0.0); asd_list.append(0.0)
        elif p.sum() == 0 or t.sum() == 0:
            hd95_list.append(100.0); asd_list.append(100.0)
        else:
            try:
                hd95_list.append(hd95(p, t))
                asd_list.append(asd(p, t))
            except:
                hd95_list.append(0.0); asd_list.append(0.0)
    hd95_val = np.mean(hd95_list)
    asd_val = np.mean(asd_list)

    pred_pixels = pred.sum().item() + 1e-7
    target_pixels = target.sum().item() + 1e-7
    area_error_percent = abs(pred_pixels - target_pixels) / target_pixels * 100

    centroid_dist_list = []
    for b in range(batch_size):
        def get_centroid(mask_tensor):
            indices = torch.nonzero(mask_tensor)
            if indices.size(0) == 0:
                return torch.tensor([0.0, 0.0], device=mask_tensor.device)
            return indices.float().mean(dim=0).flip(0)
        pred_c = get_centroid(pred[b, 0])
        target_c = get_centroid(target[b, 0])
        centroid_dist_list.append(torch.norm(pred_c - target_c).item())
    centroid_dist = np.mean(centroid_dist_list)

    boundary_f1_val = boundary_f1(pred.squeeze(), target.squeeze())

    return {
        'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall,
        'accuracy': acc, 'f1': f1, 'hd95': hd95_val, 'asd': asd_val,
        'area_error_percent': area_error_percent,
        'centroid_distance_px': centroid_dist,
        'boundary_f1': boundary_f1_val
    }

# ============ Helpers ============
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_inference_speed(model, device, num_runs=50, batch_size=1):
    device = torch.device(device)
    model.eval()
    H, W = CONFIG['image_size']
    dummy = torch.randn(batch_size, 3, H, W).to(device)
    for _ in range(10):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    total = time.time() - start
    return {'fps': num_runs * batch_size / total, 'latency_ms': total / num_runs * 1000}

def plot_curves(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['loss', 'iou', 'dice', 'hd95', 'asd', 'area_error_percent', 'centroid_distance_px', 'boundary_f1']
    titles = ['Loss', 'IoU', 'Dice', 'HD95', 'ASD', 'Area Error %', 'Centroid Dist (px)', 'Boundary F1']
    available = []
    for m in metrics:
        if f'train_{m}' in history:
            available.append(m)
    if not available:
        print("Warning: No training metrics found.")
        return
    n = len(available)
    rows = (n + 2) // 3
    cols = min(3, n)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes.flatten()
    else:
        axes = axes.flatten()
    for i, m in enumerate(available):
        ax = axes[i]
        ax.plot(history[f'train_{m}'], label='Train')
        if f'val_{m}' in history:
            ax.plot(history[f'val_{m}'], label='Val')
        ax.set_title(titles[metrics.index(m)])
        ax.legend(); ax.grid(True)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curves.png'), dpi=300)
    plt.close()

def save_diff_map(pred_bin, mask, save_path):
    diff_map = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
    fp = (pred_bin == 1) & (mask == 0)
    fn = (pred_bin == 0) & (mask == 1)
    tp = (pred_bin == 1) & (mask == 1)
    diff_map[fp] = [255, 0, 0]
    diff_map[fn] = [0, 0, 255]
    diff_map[tp] = [0, 255, 0]
    plt.imsave(save_path, diff_map)

def save_predictions(model, loader, device, output_dir, num_samples=4):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (images, masks, names) in enumerate(loader):
            if idx >= num_samples:
                break
            images = images.to(device)
            logits = model(images)
            pred = torch.sigmoid(logits).cpu().numpy()
            for i in range(min(images.size(0), 4)):
                img = images[i].cpu().numpy().transpose(1,2,0)
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                img = img * std + mean
                img = np.clip(img, 0, 1)
                mask = masks[i].cpu().numpy().squeeze()
                pred_i = pred[i].squeeze()
                pred_bin = (pred_i > 0.5).astype(np.float32)
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
                axes[1].imshow(mask, cmap='gray'); axes[1].set_title('GT'); axes[1].axis('off')
                axes[2].imshow(pred_bin, cmap='gray'); axes[2].set_title('Pred'); axes[2].axis('off')
                axes[3].imshow(img); axes[3].imshow(pred_bin, alpha=0.3, cmap='jet')
                axes[3].set_title('Overlay'); axes[3].axis('off')
                diff_path = os.path.join(output_dir, f'diff_{idx}_{i}.png')
                save_diff_map(pred_bin, mask, diff_path)
                diff_img = plt.imread(diff_path)
                axes[4].imshow(diff_img)
                axes[4].set_title('Diff (G=TP, R=FP, B=FN)'); axes[4].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'sample_{idx}_{i}.png'), dpi=150)
                plt.close()

# ============ Robustness tests ============
def test_translation_robustness(model, val_loader, device, shift_percent=0.2):
    model.eval()
    area_orig_list, area_shift_list = [], []
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            logits_orig = model(images)
            pred_orig = (torch.sigmoid(logits_orig) > 0.5).float()
            area_orig = pred_orig.sum(dim=(1,2,3)).cpu().numpy()
            _, _, H, W = images.shape
            shift_x = int(W * shift_percent * (random.random() * 2 - 1))
            shift_y = int(H * shift_percent * (random.random() * 2 - 1))
            
            h_shifted = torch.zeros_like(images)
            if shift_x >= 0:
                h_shifted[:, :, :, shift_x:] = images[:, :, :, :W-shift_x]
            else:
                h_shifted[:, :, :, :W+shift_x] = images[:, :, :, -shift_x:]
            v_shifted = torch.zeros_like(images)
            if shift_y >= 0:
                v_shifted[:, :, shift_y:, :] = h_shifted[:, :, :H-shift_y, :]
            else:
                v_shifted[:, :, :H+shift_y, :] = h_shifted[:, :, -shift_y:, :]
            shifted = v_shifted
            
            logits_shift = model(shifted)
            pred_shift = (torch.sigmoid(logits_shift) > 0.5).float()
            area_shift = pred_shift.sum(dim=(1,2,3)).cpu().numpy()
            area_orig_list.extend(area_orig.tolist())
            area_shift_list.extend(area_shift.tolist())
    area_orig = np.array(area_orig_list)
    area_shift = np.array(area_shift_list)
    rel_change = np.abs(area_shift - area_orig) / (area_orig + 1e-7)
    return {'mean_rel_area_change': np.mean(rel_change), 'std_rel_area_change': np.std(rel_change)}

def test_illumination_robustness(model, val_loader, device):
    model.eval()
    all_areas = []
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            images_unnorm = images * 0.5 + 0.5
            images_unnorm = torch.clamp(images_unnorm, 0, 1)
            logits = model(images)
            area = (torch.sigmoid(logits) > 0.5).sum(dim=(1,2,3)).cpu().numpy()
            all_areas.extend(area.tolist())
            light = torch.clamp(images_unnorm * 1.3, 0, 1)
            light_norm = (light - 0.5) / 0.5
            logits_light = model(light_norm)
            area_light = (torch.sigmoid(logits_light) > 0.5).sum(dim=(1,2,3)).cpu().numpy()
            all_areas.extend(area_light.tolist())
            dark = torch.clamp(images_unnorm * 0.8, 0, 1)
            dark_norm = (dark - 0.5) / 0.5
            logits_dark = model(dark_norm)
            area_dark = (torch.sigmoid(logits_dark) > 0.5).sum(dim=(1,2,3)).cpu().numpy()
            all_areas.extend(area_dark.tolist())
    all_areas = np.array(all_areas)
    cv = np.std(all_areas) / (np.mean(all_areas) + 1e-7)
    return {'cv': cv, 'mean_area': np.mean(all_areas), 'std_area': np.std(all_areas)}

def test_scale_robustness(model, val_loader, device, scales=[0.8, 0.9, 1.0, 1.1, 1.2]):
    model.eval()
    results = {scale: [] for scale in scales}
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            for scale in scales:
                scaled_size = (int(images.shape[2]*scale), int(images.shape[3]*scale))
                scaled_images = F.interpolate(images, size=scaled_size, mode='bilinear', align_corners=False)
                scaled_masks = F.interpolate(masks, size=scaled_size, mode='nearest')
                logits = model(scaled_images)
                pred = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred * scaled_masks).sum()
                union = (pred + scaled_masks - pred * scaled_masks).sum()
                iou = (inter / (union + 1e-7)).item()
                results[scale].append(iou)
    return {scale: np.mean(ious) for scale, ious in results.items()}

# ============ generate_params_table covers only the models passed in ============
def generate_params_table(models_dict):
    """
    Takes a dict of {model name: model instance} and returns a DataFrame of
    parameter counts.
    """
    table = []
    for name, model in models_dict.items():
        params = count_parameters(model)
        table.append({'Model': name, 'Params (M)': f'{params:.2f}'})
    return pd.DataFrame(table)

# ============ Training, with timing ============
def train_one_epoch(model, train_loader, val_loader, fold, output_dir):
    device = CONFIG['device']
    writer = SummaryWriter(log_dir=os.path.join(output_dir, f'fold_{fold}', 'logs')) if CONFIG['use_tensorboard'] else None

    criterion = DiceBCELoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = amp.GradScaler()

    best_iou = 0.0
    best_dice = 0.0
    patience_counter = 0
    best_val_loss = float('inf')

    history_keys = ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                    'train_dice', 'val_dice', 'train_hd95', 'val_hd95',
                    'train_asd', 'val_asd',
                    'train_area_error_percent', 'val_area_error_percent',
                    'train_centroid_distance_px', 'val_centroid_distance_px',
                    'train_boundary_f1', 'val_boundary_f1']
    history = {k: [] for k in history_keys}

    fold_dir = os.path.join(output_dir, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # ===== start of training =====
    train_start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        # ---------- train ----------
        model.train()
        train_loss, train_iou, train_dice = 0, 0, 0
        train_hd95, train_asd = 0, 0
        train_area_err, train_centroid, train_boundary = 0, 0, 0

        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            m = calc_metrics(logits, masks)
            train_iou += m['iou'] * images.size(0)
            train_dice += m['dice'] * images.size(0)
            train_hd95 += m['hd95'] * images.size(0)
            train_asd += m['asd'] * images.size(0)
            train_area_err += m['area_error_percent'] * images.size(0)
            train_centroid += m['centroid_distance_px'] * images.size(0)
            train_boundary += m['boundary_f1'] * images.size(0)

        n_train = len(train_loader.dataset)
        train_loss /= n_train; train_iou /= n_train; train_dice /= n_train
        train_hd95 /= n_train; train_asd /= n_train
        train_area_err /= n_train; train_centroid /= n_train; train_boundary /= n_train

        # ---------- validate ----------
        model.eval()
        val_loss, val_iou, val_dice = 0, 0, 0
        val_hd95, val_asd = 0, 0
        val_area_err, val_centroid, val_boundary = 0, 0, 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
                m = calc_metrics(logits, masks)
                val_iou += m['iou'] * images.size(0)
                val_dice += m['dice'] * images.size(0)
                val_hd95 += m['hd95'] * images.size(0)
                val_asd += m['asd'] * images.size(0)
                val_area_err += m['area_error_percent'] * images.size(0)
                val_centroid += m['centroid_distance_px'] * images.size(0)
                val_boundary += m['boundary_f1'] * images.size(0)

        n_val = len(val_loader.dataset)
        val_loss /= n_val; val_iou /= n_val; val_dice /= n_val
        val_hd95 /= n_val; val_asd /= n_val
        val_area_err /= n_val; val_centroid /= n_val; val_boundary /= n_val

        scheduler.step(val_loss)

        hist_vals = [train_loss, val_loss, train_iou, val_iou, train_dice, val_dice,
                     train_hd95, val_hd95, train_asd, val_asd,
                     train_area_err, val_area_err, train_centroid, val_centroid,
                     train_boundary, val_boundary]
        for key, val in zip(history_keys, hist_vals):
            history[key].append(val)
            if writer:
                writer.add_scalar(key, val, epoch)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_iou.pth'))
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_dice.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"[Epoch {epoch+1:2d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
              f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
              f"Val HD95: {val_hd95:.4f} | Val ASD: {val_asd:.4f} | "
              f"AreaErr: {val_area_err:.2f}% | BoundaryF1: {val_boundary:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if writer:
        writer.close()

    # ===== total time for this fold =====
    train_total_time = time.time() - train_start_time
    train_time_min = train_total_time / 60.0

    # ---------- test after training ----------
    print("\nRunning robustness tests...")
    illum_results = test_illumination_robustness(model, val_loader, device)
    trans_results = test_translation_robustness(model, val_loader, device, shift_percent=0.2)
    scale_results = test_scale_robustness(model, val_loader, device)

    save_predictions(model, val_loader, device, os.path.join(fold_dir, 'predictions'))
    plot_curves(history, fold_dir)
    pd.DataFrame(history).to_csv(os.path.join(fold_dir, 'history.csv'), index=False)
    speed = measure_inference_speed(model, device)

    results = {
        'fold': fold,
        'best_iou': best_iou,
        'best_dice': best_dice,
        'params_M': count_parameters(model),
        'fps': speed['fps'],
        'latency_ms': speed['latency_ms'],
        'train_time_min': train_time_min,
        'illumination_cv': illum_results['cv'],
        'translation_mean_area_change': trans_results['mean_rel_area_change'],
        'scale_iou_0.8': scale_results[0.8],
        'scale_iou_0.9': scale_results[0.9],
        'scale_iou_1.0': scale_results[1.0],
        'scale_iou_1.1': scale_results[1.1],
        'scale_iou_1.2': scale_results[1.2],
        **{f'val_{k}': history[f'val_{k}'][-1] for k in ['iou', 'dice', 'hd95', 'asd',
                                                         'area_error_percent',
                                                         'centroid_distance_px',
                                                         'boundary_f1']}
    }
    return results

# ============ Main ============
def main():
    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)

    # ===== parameter-count table over the models actually defined =====
    models_to_compare = {
        'UNet-FreqMamba (ours)': UNet_FreqMamba(base=64),
        # further models can be added here once they are defined
    }
    params_df = generate_params_table(models_to_compare)
    params_df.to_csv(os.path.join(output_dir, 'params_comparison.csv'), index=False)
    print("\nParameter-count comparison:")
    print(params_df.to_string(index=False))

    # load the full dataset
    full_dataset = ShellDataset(CONFIG['image_dir'], CONFIG['mask_dir'], CONFIG['image_size'])
    if CONFIG['subset_size'] and CONFIG['subset_size'] > 0:
        full_dataset = create_subset(full_dataset, CONFIG['subset_size'], CONFIG['seed'])
    print(f"Full dataset size: {len(full_dataset)}")

    # hold out the test set
    test_ratio = CONFIG.get('test_ratio', 0.15)
    total_len = len(full_dataset)
    test_len = int(total_len * test_ratio)
    trainval_len = total_len - test_len
    trainval_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [trainval_len, test_len],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    print(f"Train+Val dataset size: {trainval_len}, Test dataset size: {test_len}")

    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'])

    # K-fold cross-validation
    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    indices = list(range(len(trainval_dataset)))
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*60}\nFold {fold+1}/{CONFIG['n_folds']}\n{'='*60}")
        train_ds = Subset(trainval_dataset, [indices[i] for i in train_idx])
        val_ds = Subset(trainval_dataset, [indices[i] for i in val_idx])
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                                num_workers=CONFIG['num_workers'])

        model = UNet_FreqMamba().to(device)
        print(f"Model params: {count_parameters(model):.2f}M")
        speed = measure_inference_speed(model, device)
        print(f"Inference: {speed['fps']:.1f} FPS, {speed['latency_ms']:.2f} ms")

        results = train_one_epoch(model, train_loader, val_loader, fold, output_dir)
        all_results.append(results)
        print(f"Fold {fold+1} Best IoU: {results['best_iou']:.4f}, Best Dice: {results['best_dice']:.4f}")

    # aggregate the cross-validation results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    print("\n" + "="*80)
    print("AVERAGE ACROSS ALL FOLDS")
    print("="*80)
    avg_cols = ['best_iou', 'best_dice', 'val_hd95', 'val_asd',
                'val_area_error_percent', 'val_centroid_distance_px',
                'val_boundary_f1', 'params_M', 'fps', 'latency_ms', 'train_time_min']
    for col in avg_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{col:35s}: {mean_val:.4f} ± {std_val:.4f}")

    # test-set evaluation, using the fold with the best validation IoU
    best_fold = -1
    best_iou_val = -1.0
    for fold in range(CONFIG['n_folds']):
        fold_iou = df[df['fold'] == fold]['best_iou'].values[0] if fold in df['fold'].values else None
        if fold_iou is not None and fold_iou > best_iou_val:
            best_iou_val = fold_iou
            best_fold = fold

    if best_fold != -1:
        best_model_path = os.path.join(output_dir, f'fold_{best_fold}', 'best_iou.pth')
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from fold {best_fold} (val IoU = {best_iou_val:.4f}) for test evaluation...")
            model = UNet_FreqMamba().to(device)
            # strict=False tolerates buffer differences
            model.load_state_dict(torch.load(best_model_path), strict=False)
            model.eval()
            test_metrics = {'iou': [], 'dice': [], 'hd95': [], 'asd': [],
                            'area_error_percent': [], 'centroid_distance_px': [], 'boundary_f1': []}
            with torch.no_grad():
                for images, masks, _ in test_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    m = calc_metrics(logits, masks)
                    for key in test_metrics:
                        test_metrics[key].append(m[key])
            test_avg = {k: np.mean(v) for k, v in test_metrics.items()}
            print("\n" + "="*80)
            print("TEST SET EVALUATION (on best model with highest val IoU across folds)")
            print("="*80)
            for k, v in test_avg.items():
                print(f"{k:25s}: {v:.4f}")
            pd.DataFrame([test_avg]).to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
        else:
            print(f"Best model file not found: {best_model_path}")
    else:
        print("No best model found.")

    print(f"\nAll results written to: {output_dir}")

if __name__ == "__main__":
    main()