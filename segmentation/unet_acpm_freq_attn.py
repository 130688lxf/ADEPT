#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 unet_acpm_freq_attn.py -- control variant with self-attention in place of Mamba
================================================================================

 This is not a reconstruction of TransUNet. That design puts attention after a
 CNN encoder that has downsampled only four times, giving 22016 tokens, which
 exhausts GPU memory and is not worth the engineering. What is done here is
 smaller and more direct: only the single position currently occupied by Mamba
 is replaced by standard self-attention, and everything else -- ACPM, the
 frequency branch FFE, the gating and the downsampling depth -- is left as it
 is.

 Mamba sits inside FreqEMBlock, after the third and fourth downsampling steps:
   freqem3: a 512x688 input downsampled 8x  -> 64x86 = 5504 tokens
   freqem4: downsampled 16x                 -> 32x43 = 1376 tokens
 This is the downsampling depth TransUNet should have used (it is the scale
 that patch_size=16 corresponds to). F.scaled_dot_product_attention, built into
 torch >= 2.0, dispatches to a memory-efficient flash-attention backend and
 never materialises the 5504x5504 attention matrix, so attention is feasible at
 this scale.

 Apart from the single substitution self.ss2d -> self.attn, everything matches
 the Mamba variant, including the three evaluation conventions shared across
 the ablation: no augmentation on validation and test, 5x5 closing plus largest
 connected component as post-processing, and illumination robustness computed
 per image. That makes this variant directly comparable with the other seven.
================================================================================
"""
import torch
torch.cuda.empty_cache()
import os
import random
import json
import time
import math
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
from scipy import ndimage as _ndi

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# ============ Configuration, identical to the other seven variants ============
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
    "output_dir": "./outputs_ACPM_Freq_Attn",
    "test_ratio": 0.15,
}

POSTPROCESS = True  # 5x5 closing + largest connected component, as in the published protocol

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

# ============ Dataset: training augmented, validation and test not ============
class ShellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512,688), augment=True):
        self.image_dir = image_dir; self.mask_dir = mask_dir
        self.image_size = image_size; self.augment = augment
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
                    self.pairs.append((img_f, mask_f)); break
        if not self.pairs:
            raise RuntimeError("No image-mask pairs found.")
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, mask_name))
        mask = mask.split()[3] if mask.mode == 'RGBA' else mask.convert('L')
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
    random.seed(seed); random.shuffle(indices)
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


# ====== The only substantive change: SelfAttention2D replaces SS2D_Radial ======
class SelfAttention2D(nn.Module):
    """
    Same input and output contract as SS2D_Radial: forward(x) takes (B,C,H,W)
    and returns x + attention_out, so it can replace self.ss2d inside
    VisualStateSpaceBlock without touching any surrounding code.

    F.scaled_dot_product_attention (built into torch >= 2.0, dispatching to a
    memory-efficient or flash-attention backend) is used rather than an explicit
    QK^T followed by softmax: at 5504 tokens the explicit form materialises a
    5504x5504 attention matrix, which is what makes this scale intractable in a
    """
    def __init__(self, dim, n_heads=8):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        qkv = self.qkv(seq).reshape(B, H * W, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, HW, head_dim)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + out


class VisualStateSpaceBlock(nn.Module):
    """Identical to the established block except that self.ss2d becomes self.attn."""
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.attn = SelfAttention2D(dim, n_heads=n_heads)   # <-- the only change
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x1 = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x1 = self.silu(self.linear1(x1))
        x2 = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x2 = self.linear2(x2)
        x2 = x2.permute(0, 2, 1).reshape(B, C, H, W)
        x2 = self.dwconv(x2)
        x2 = self.silu(x2)
        x2 = self.attn(x2)                                  # <-- the only change
        x2 = x2.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x2 = self.norm(x2)
        out = x1 * x2
        out = self.out_linear(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + out


class FreqEMBlock(nn.Module):
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.vss = VisualStateSpaceBlock(channels // 2, n_heads=n_heads)
        self.ffe = FrequencyEnhancementModule(channels // 2)
        self.gate_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.center_predictor = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.sigma_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 1), nn.Sigmoid())

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
        x = x.permute(0, 2, 3, 1); x = self.norm(x); x = x.permute(0, 3, 1, 2)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.vss(x1)
        x2 = self.ffe(x2)
        x_cat = torch.cat([x1, x2], dim=1)
        x_cat = identity + x_cat
        gate = torch.sigmoid(self.gate_conv(x_cat))
        x_cat = x_cat * (1 + gate)
        x_cat = x_cat.permute(0, 2, 3, 1); x_cat = self.norm(x_cat); x_cat = x_cat.permute(0, 3, 1, 2)
        x_cat = self.dwconv(x_cat); x_cat = self.bn(x_cat); x_cat = self.relu(x_cat)
        return x_cat + identity


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = nn.MaxPool2d(2); self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.mp(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet_ACPM_FreqAttn(nn.Module):
    """Control variant of UNet_FreqMamba: layer for layer identical, Mamba replaced by attention."""
    def __init__(self, base=64):
        super().__init__()
        self.inc = DoubleConv(3, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.freqem3 = FreqEMBlock(base*8)     # 1/8 resolution, 5504 tokens
        self.down4 = Down(base*8, base*16)
        self.freqem4 = FreqEMBlock(base*16)    # 1/16 resolution, 1376 tokens
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
        x = self.up1(x5, x4); x = self.up2(x, x3)
        x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)


# ============ Loss, post-processing and metrics ============
class DiceBCELoss(nn.Module):
    def __init__(self, alpha=0.5, eps=1e-7):
        super().__init__(); self.alpha = alpha; self.eps = eps
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum()
        dice = 1 - (2*inter + self.eps) / (pred.sum() + target.sum() + self.eps)
        return self.alpha * bce + (1 - self.alpha) * dice


def postprocess_mask(pred):
    if not POSTPROCESS:
        return pred
    device = pred.device
    out = pred.detach().cpu().numpy()
    st = np.ones((5, 5), dtype=bool)
    for b in range(out.shape[0]):
        m = out[b, 0] > 0.5
        if m.sum() == 0:
            continue
        m = _ndi.binary_closing(m, structure=st)
        lab, n = _ndi.label(m)
        if n > 1:
            sizes = _ndi.sum(m, lab, range(1, n + 1))
            m = (lab == (int(np.argmax(sizes)) + 1))
        out[b, 0] = m.astype(np.float32)
    return torch.from_numpy(out).to(device)


def boundary_f1(pred, target, threshold=3):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0); target = target.unsqueeze(0)
    pred = pred.float(); target = target.float()
    kernel_size = 2 * threshold + 1; padding = threshold
    pred_pad = F.pad(pred, (padding,)*4, mode='constant', value=0)
    target_pad = F.pad(target, (padding,)*4, mode='constant', value=0)
    pred_dilated = F.max_pool2d(pred_pad, kernel_size, stride=1, padding=0)
    target_dilated = F.max_pool2d(target_pad, kernel_size, stride=1, padding=0)
    pred_boundary = (pred_dilated - pred > 0.5).float()
    target_boundary = (target_dilated - target > 0.5).float()
    tp = (pred_boundary * target_boundary).sum(dim=(1,2))
    fp = (pred_boundary * (1 - target_boundary)).sum(dim=(1,2))
    fn = ((1 - pred_boundary) * target_boundary).sum(dim=(1,2))
    precision = tp / (tp + fp + 1e-7); recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1.mean().item()


def calc_metrics(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    pred = postprocess_mask(pred)
    target = target.float()
    batch_size = pred.shape[0]
    iou_sum, dice_sum = 0.0, 0.0
    for b in range(batch_size):
        p = pred[b,0]; t = target[b,0]
        inter = (p*t).sum(); union = (p+t-p*t).sum()+1e-7
        iou_sum += (inter/union).item()
        dice_sum += (2*inter/(p.sum()+t.sum()+1e-7)).item()
    iou = iou_sum/batch_size; dice = dice_sum/batch_size
    pred_np = pred.cpu().numpy().flatten().astype(int)
    target_np = target.cpu().numpy().flatten().astype(int)
    tn,fp,fn,tp = confusion_matrix(target_np, pred_np, labels=[0,1]).ravel()
    precision = tp/(tp+fp+1e-7); recall = tp/(tp+fn+1e-7)
    acc = (tp+tn)/(tp+tn+fp+fn+1e-7); f1 = 2*precision*recall/(precision+recall+1e-7)
    hd95_list, asd_list = [], []
    for b in range(batch_size):
        p = pred[b,0].cpu().numpy().astype(np.bool_)
        t = target[b,0].cpu().numpy().astype(np.bool_)
        if p.sum()==0 and t.sum()==0:
            hd95_list.append(0.0); asd_list.append(0.0)
        elif p.sum()==0 or t.sum()==0:
            hd95_list.append(100.0); asd_list.append(100.0)
        else:
            try: hd95_list.append(hd95(p,t)); asd_list.append(asd(p,t))
            except: hd95_list.append(0.0); asd_list.append(0.0)
    hd95_val = np.mean(hd95_list); asd_val = np.mean(asd_list)
    pred_pixels = pred.sum().item()+1e-7; target_pixels = target.sum().item()+1e-7
    area_error_percent = abs(pred_pixels-target_pixels)/target_pixels*100
    centroid_dist_list = []
    for b in range(batch_size):
        def get_centroid(mt):
            idx = torch.nonzero(mt)
            return (torch.tensor([0.,0.],device=mt.device) if idx.size(0)==0
                   else idx.float().mean(dim=0).flip(0))
        pred_c = get_centroid(pred[b,0]); target_c = get_centroid(target[b,0])
        centroid_dist_list.append(torch.norm(pred_c-target_c).item())
    centroid_dist = np.mean(centroid_dist_list)
    boundary_f1_val = boundary_f1(pred.squeeze(), target.squeeze())
    return {'iou':iou,'dice':dice,'precision':precision,'recall':recall,'accuracy':acc,
            'f1':f1,'hd95':hd95_val,'asd':asd_val,'area_error_percent':area_error_percent,
            'centroid_distance_px':centroid_dist,'boundary_f1':boundary_f1_val}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_inference_speed(model, device, num_runs=50, batch_size=1):
    device = torch.device(device); model.eval()
    H, W = CONFIG['image_size']
    dummy = torch.randn(batch_size, 3, H, W).to(device)
    for _ in range(10): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs): _ = model(dummy)
    if device.type == 'cuda': torch.cuda.synchronize()
    total = time.time() - start
    return {'fps': num_runs*batch_size/total, 'latency_ms': total/num_runs*1000}


def test_illumination_robustness(model, val_loader, device):
    model.eval()
    per_image_cv, rel_bright, rel_dark, areas_orig = [], [], [], []
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            images_unnorm = torch.clamp(images * 0.5 + 0.5, 0, 1)
            def _area(x_norm):
                lg = model(x_norm); pr = (torch.sigmoid(lg) > 0.5).float()
                pr = postprocess_mask(pr)
                return pr.sum(dim=(1,2,3)).cpu().numpy()
            a0 = _area(images)
            light = torch.clamp(images_unnorm * 1.3, 0, 1)
            a1 = _area((light - 0.5) / 0.5)
            dark = torch.clamp(images_unnorm * 0.8, 0, 1)
            a2 = _area((dark - 0.5) / 0.5)
            trio = np.stack([a0, a1, a2], axis=1)
            per_image_cv.extend((trio.std(axis=1)/(trio.mean(axis=1)+1e-7)).tolist())
            rel_bright.extend((np.abs(a1-a0)/(a0+1e-7)).tolist())
            rel_dark.extend((np.abs(a2-a0)/(a0+1e-7)).tolist())
            areas_orig.extend(a0.tolist())
    return {'cv': float(np.mean(per_image_cv)),
            'mean_rel_change_bright': float(np.mean(rel_bright)),
            'mean_rel_change_dark': float(np.mean(rel_dark)),
            'mean_area': float(np.mean(areas_orig)), 'std_area': float(np.std(areas_orig))}


def test_translation_robustness(model, val_loader, device, shift_percent=0.2):
    model.eval(); area_orig_list, area_shift_list = [], []
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images = images.to(device)
            logits_orig = model(images)
            pred_orig = postprocess_mask((torch.sigmoid(logits_orig) > 0.5).float())
            area_orig = pred_orig.sum(dim=(1,2,3)).cpu().numpy()
            _, _, H, W = images.shape
            shift_x = int(W*shift_percent*(random.random()*2-1))
            shift_y = int(H*shift_percent*(random.random()*2-1))
            h_shifted = torch.zeros_like(images)
            if shift_x >= 0: h_shifted[:,:,:,shift_x:] = images[:,:,:,:W-shift_x]
            else: h_shifted[:,:,:,:W+shift_x] = images[:,:,:,-shift_x:]
            v_shifted = torch.zeros_like(images)
            if shift_y >= 0: v_shifted[:,:,shift_y:,:] = h_shifted[:,:,:H-shift_y,:]
            else: v_shifted[:,:,:H+shift_y,:] = h_shifted[:,:,-shift_y:,:]
            logits_shift = model(v_shifted)
            pred_shift = postprocess_mask((torch.sigmoid(logits_shift) > 0.5).float())
            area_shift = pred_shift.sum(dim=(1,2,3)).cpu().numpy()
            area_orig_list.extend(area_orig.tolist()); area_shift_list.extend(area_shift.tolist())
    area_orig = np.array(area_orig_list); area_shift = np.array(area_shift_list)
    rel_change = np.abs(area_shift-area_orig)/(area_orig+1e-7)
    return {'mean_rel_area_change': np.mean(rel_change), 'std_rel_area_change': np.std(rel_change)}


def test_scale_robustness(model, val_loader, device, scales=[0.8,0.9,1.0,1.1,1.2]):
    model.eval(); results = {s: [] for s in scales}
    with torch.no_grad():
        for images, masks, _ in val_loader:
            images, masks = images.to(device), masks.to(device)
            for scale in scales:
                sz = (int(images.shape[2]*scale), int(images.shape[3]*scale))
                si = F.interpolate(images, size=sz, mode='bilinear', align_corners=False)
                sm = F.interpolate(masks, size=sz, mode='nearest')
                pred = postprocess_mask((torch.sigmoid(model(si)) > 0.5).float())
                inter = (pred*sm).sum(); union = (pred+sm-pred*sm).sum()
                results[scale].append((inter/(union+1e-7)).item())
    return {s: np.mean(v) for s, v in results.items()}


def train_one_epoch(model, train_loader, val_loader, fold, output_dir):
    device = CONFIG['device']
    writer = SummaryWriter(log_dir=os.path.join(output_dir, f'fold_{fold}', 'logs')) \
        if CONFIG['use_tensorboard'] else None
    criterion = DiceBCELoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                            weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = amp.GradScaler()
    best_iou = 0.0; best_dice = 0.0; patience_counter = 0; best_val_loss = float('inf')
    history_keys = ['train_loss','val_loss','train_iou','val_iou','train_dice','val_dice',
                    'train_hd95','val_hd95','train_asd','val_asd',
                    'train_area_error_percent','val_area_error_percent',
                    'train_centroid_distance_px','val_centroid_distance_px',
                    'train_boundary_f1','val_boundary_f1']
    history = {k: [] for k in history_keys}
    fold_dir = os.path.join(output_dir, f'fold_{fold}'); os.makedirs(fold_dir, exist_ok=True)
    train_start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_loss=train_iou=train_dice=train_hd95=train_asd=0
        train_area_err=train_centroid=train_boundary=0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                logits = model(images); loss = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item()*images.size(0)
            m = calc_metrics(logits, masks)
            train_iou += m['iou']*images.size(0); train_dice += m['dice']*images.size(0)
            train_hd95 += m['hd95']*images.size(0); train_asd += m['asd']*images.size(0)
            train_area_err += m['area_error_percent']*images.size(0)
            train_centroid += m['centroid_distance_px']*images.size(0)
            train_boundary += m['boundary_f1']*images.size(0)
        n_train = len(train_loader.dataset)
        train_loss/=n_train; train_iou/=n_train; train_dice/=n_train
        train_hd95/=n_train; train_asd/=n_train
        train_area_err/=n_train; train_centroid/=n_train; train_boundary/=n_train

        model.eval()
        val_loss=val_iou=val_dice=val_hd95=val_asd=0
        val_area_err=val_centroid=val_boundary=0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images); loss = criterion(logits, masks)
                val_loss += loss.item()*images.size(0)
                m = calc_metrics(logits, masks)
                val_iou += m['iou']*images.size(0); val_dice += m['dice']*images.size(0)
                val_hd95 += m['hd95']*images.size(0); val_asd += m['asd']*images.size(0)
                val_area_err += m['area_error_percent']*images.size(0)
                val_centroid += m['centroid_distance_px']*images.size(0)
                val_boundary += m['boundary_f1']*images.size(0)
        n_val = len(val_loader.dataset)
        val_loss/=n_val; val_iou/=n_val; val_dice/=n_val
        val_hd95/=n_val; val_asd/=n_val
        val_area_err/=n_val; val_centroid/=n_val; val_boundary/=n_val
        scheduler.step(val_loss)
        hist_vals = [train_loss,val_loss,train_iou,val_iou,train_dice,val_dice,
                    train_hd95,val_hd95,train_asd,val_asd,train_area_err,val_area_err,
                    train_centroid,val_centroid,train_boundary,val_boundary]
        for key, val in zip(history_keys, hist_vals):
            history[key].append(val)
            if writer: writer.add_scalar(key, val, epoch)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_iou.pth'))
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_dice.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_counter = 0
        else:
            patience_counter += 1
        print(f"[Epoch {epoch+1:2d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
              f"Val HD95: {val_hd95:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}"); break
    if writer: writer.close()
    train_time_min = (time.time() - train_start_time) / 60.0

    print("\nRunning robustness tests...")
    illum_results = test_illumination_robustness(model, val_loader, device)
    trans_results = test_translation_robustness(model, val_loader, device, shift_percent=0.2)
    scale_results = test_scale_robustness(model, val_loader, device)
    pd.DataFrame(history).to_csv(os.path.join(fold_dir, 'history.csv'), index=False)
    speed = measure_inference_speed(model, device)
    results = {'fold': fold, 'best_iou': best_iou, 'best_dice': best_dice,
              'params_M': count_parameters(model), 'fps': speed['fps'],
              'latency_ms': speed['latency_ms'], 'train_time_min': train_time_min,
              'illumination_cv': illum_results['cv'],
              'illum_rel_change_bright': illum_results['mean_rel_change_bright'],
              'illum_rel_change_dark': illum_results['mean_rel_change_dark'],
              'translation_mean_area_change': trans_results['mean_rel_area_change'],
              'scale_iou_0.8': scale_results[0.8], 'scale_iou_0.9': scale_results[0.9],
              'scale_iou_1.0': scale_results[1.0], 'scale_iou_1.1': scale_results[1.1],
              'scale_iou_1.2': scale_results[1.2],
              **{f'val_{k}': history[f'val_{k}'][-1] for k in
                 ['iou','dice','hd95','asd','area_error_percent',
                  'centroid_distance_px','boundary_f1']}}
    return results


def main():
    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(CONFIG['output_dir'], f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)

    m = UNet_ACPM_FreqAttn(base=64)
    print(f"parameters: {count_parameters(m):.2f}M")

    full_dataset = ShellDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                                CONFIG['image_size'], augment=True)
    full_dataset_eval = ShellDataset(CONFIG['image_dir'], CONFIG['mask_dir'],
                                     CONFIG['image_size'], augment=False)
    if CONFIG['subset_size']:
        full_dataset = create_subset(full_dataset, CONFIG['subset_size'], CONFIG['seed'])
        full_dataset_eval = create_subset(full_dataset_eval, CONFIG['subset_size'], CONFIG['seed'])
    total_len = len(full_dataset)
    test_len = int(total_len * CONFIG['test_ratio'])
    trainval_len = total_len - test_len
    _perm = torch.randperm(total_len, generator=torch.Generator().manual_seed(
        CONFIG['seed'])).tolist()
    trainval_indices = _perm[:trainval_len]; test_indices = _perm[trainval_len:]
    trainval_dataset = Subset(full_dataset, trainval_indices)
    trainval_dataset_eval = Subset(full_dataset_eval, trainval_indices)
    test_dataset = Subset(full_dataset_eval, test_indices)
    print(f"Train+Val: {trainval_len}, Test: {test_len}  [augment] train=True val=False test=False")
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=CONFIG['num_workers'])

    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    indices = list(range(len(trainval_dataset)))
    all_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n{'='*60}\nFold {fold+1}/{CONFIG['n_folds']}\n{'='*60}")
        train_ds = Subset(trainval_dataset, [indices[i] for i in train_idx])
        val_ds = Subset(trainval_dataset_eval, [indices[i] for i in val_idx])
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                  num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                                num_workers=CONFIG['num_workers'])
        model = UNet_ACPM_FreqAttn().to(device)
        results = train_one_epoch(model, train_loader, val_loader, fold, output_dir)
        all_results.append(results)
        print(f"Fold {fold+1} Best IoU: {results['best_iou']:.4f}")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    print("\n" + "="*80); print(df.to_string(index=False))

    best_fold = int(df.loc[df.best_iou.idxmax(), 'fold'])
    best_model_path = os.path.join(output_dir, f'fold_{best_fold}', 'best_iou.pth')
    model = UNet_ACPM_FreqAttn().to(device)
    model.load_state_dict(torch.load(best_model_path), strict=False)
    model.eval()
    test_metrics = {k: [] for k in ['iou','dice','hd95','asd','area_error_percent',
                                    'centroid_distance_px','boundary_f1']}
    with torch.no_grad():
        for images, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)
            m = calc_metrics(model(images), masks)
            for k in test_metrics: test_metrics[k].append(m[k])
    test_avg = {k: np.mean(v) for k, v in test_metrics.items()}
    pd.DataFrame([test_avg]).to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    print(f"\nAll results written to: {output_dir}")


if __name__ == "__main__":
    main()
