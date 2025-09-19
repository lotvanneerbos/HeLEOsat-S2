#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegFormer Super-Resolution (training + evaluation)

What this script does
- Trains a SegFormer-based regressor to super-resolve low-resolution GHI tiles into high-resolution targets.
- Pairs LR↔HR scenes by acquisition date, samples random crops per epoch, and evaluates on val/test.
- Saves model checkpoint, figures (loss / histogram / scatter / PSD), arrays, and a CSV with metrics.

Inputs on disk (fill in with your own paths)
- base_lr_path/
    ├── train/lr/*.tif
    ├── val/lr/*.tif      (fallback: val_all/*.tif)
    └── test/lr/*.tif     (fallback: test_all/*.tif)
- base_hr_path/
    ├── train/hr/*.tif
    ├── val/hr/*.tif      (fallback: val_all/*.tif)
    └── test/hr/*.tif     (fallback: test_all/*.tif)

Outputs
- Run folder under out_root with:
  • model checkpoint (.pth)
  • loss.(png|pdf), histogram.(png|pdf), density_scatter.(png|pdf), scatter.(png|pdf), psd.(png|pdf)
  • psd_*.npy arrays
  • evaluation_metrics.csv
  • eval_config.json

Requirements
- Python 3.9+
- pip install: torch, torchvision, transformers, numpy, pandas, rasterio, pillow, matplotlib, scipy

Usage (quick start)
-------------------
# 1) Edit DEFAULTS below (paths, hyperparameters).
# 2) Run:  python segformer_train_validate.py
#    The __main__ block calls train_val_test_model(**DEFAULTS).
"""

# =======================
# Defaults (edit these)
# =======================
DEFAULTS = {
    "base_lr_path": "/path_to_lr_root",
    "base_hr_path": "/path_to_hr_root",

    # Train subset root (set None to use standard train/lr & train/hr)
    # Examples:
    #   "/path_to_lr_root/random_301/1"  (and matching HR path)
    #   "/path_to_subset_root" containing subfolders "lr" and "hr"
    "train_subset_root": None,  # or "/path_to_train_subset"

    "resolution_meters": 40,
    "weights": "imagenet",      # "imagenet" or "none"
    "lr": 1e-4,
    "num_epochs": 30,
    "batch_size": 8,
    "tile_size": 512,
    "train_tiles": 1000,
    "val_tiles": 200,
    "test_tiles": 200,
    "num_workers": 4,

    "out_root": "/path_to_output_models/segformer_b0",
    "loss_ylim": None,          # or (ymin, ymax)

    # If you leave these as None, code falls back to .../val/lr or val_all, .../test/lr or test_all
    "val_lr_dir": None,
    "val_hr_dir": None,
    "test_lr_dir": None,
    "test_hr_dir": None,
}

# =======================
# Figure style (thesis-quality)
# =======================
import os, re, json, csv, math, time, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import rasterio
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless / HPC
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 600, "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})

# =======================
# Evaluation config (consistent plots/units)
# =======================
EVAL_CFG = {
    # If you normalise to [0,1] during training: set True and provide physical scale
    "eval_in_physical_units": True,
    "ghi_scale_max": 1200.0,    # scale used in dataset normalisation
    "unit_in":  "W/m^2",        # set "Wh/m^2" if your inputs are energy per minute
    "unit_out": "W/m^2",
    "dt_minutes": 1.0,

    # Histogram
    "hist_bins": 120,
    "hist_range_wh": (0.0, 20.0),
    "hist_range_w":  (0.0, 1200.0),

    # Scatter/hexbin
    "hexbin_gridsize": 120,
    "scatter_range_wh": (0.0, 20.0),
    "scatter_range_w":  (0.0, 1200.0),

    # PSD
    # fs=1.0 means per-pixel frequency. For spatial frequencies in m^-1 at 40 m/px, use fs=1/40.
    "psd_fs": 1.0,
    "psd_y_fixed": False,
    "psd_y_min": 1e-10,
    "psd_y_max": 1e0,
    "psd_y_percentile_clip": (1, 99),
    "psd_bins": [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, None)],
}

# =======================
# Metrics helpers
# =======================
def mse(a, b): return float(np.mean((a - b) ** 2))
def mae(a, b): return float(np.mean(np.abs(a - b)))
def mbe(a, b): return float(np.mean(a - b))
def corr_pearson(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# =======================
# PSD metrics: radial PSD and band correlations
# =======================
def compute_psd_metrics(gt, pred, fs=1.0, bins_cfg=((0.0,0.05),(0.05,0.15),(0.15,0.3),(0.3,None))):
    gt = np.asarray(gt); pred = np.asarray(pred)
    if gt.ndim == 3:
        _, H, W = gt.shape
    elif gt.ndim == 2:
        H, W = gt.shape
    else:
        raise ValueError("gt/pred must be 2D (H,W) or 3D (N,H,W)")

    def _psd2(x):
        X = np.fft.rfft2(x)
        return (np.abs(X) ** 2) / (x.size)

    if gt.ndim == 3:
        Pgt = np.mean([_psd2(x) for x in gt], axis=0)
        Ppr = np.mean([_psd2(x) for x in pred], axis=0)
    else:
        Pgt = _psd2(gt); Ppr = _psd2(pred)

    d  = 1.0 / fs
    fy = np.fft.fftfreq(H, d=d)
    fx = np.fft.rfftfreq(W, d=d)
    Fy, Fx = np.meshgrid(fy, fx, indexing="ij")
    fr = np.sqrt(Fx**2 + Fy**2)

    r = fr.ravel(); pg = Pgt.ravel(); pp = Ppr.ravel()
    order = np.argsort(r)
    r_sorted = r[order]; pg_sorted = pg[order]; pp_sorted = pp[order]

    nb = 400
    idxs = np.linspace(0, len(r_sorted)-1, nb).astype(int)
    f_plot   = r_sorted[idxs]
    Pgt_plot = pg_sorted[idxs]
    Ppr_plot = pp_sorted[idxs]

    def _corr(x, y):
        xs, ys = x.std(), y.std()
        if xs == 0 or ys == 0: return 0.0
        return float(np.corrcoef(x, y)[0,1])

    bin_corr = []
    rmax = r.max()
    for lo, hi in bins_cfg:
        lo = 0.0 if lo is None else lo
        hi = rmax if hi is None else hi
        mask = (r >= lo) & (r < hi)
        if mask.sum() < 10:
            bin_corr.append(0.0)
        else:
            bin_corr.append(_corr(pg[mask], pp[mask]))

    return f_plot, Pgt_plot, Ppr_plot, bin_corr

# =======================
# Dataset
# =======================
def extract_first_date(filename):
    m = re.findall(r"20\d{6}", filename)  # matches YYYYMMDD starting with '20'
    return m[0] if m else None

class GHIDownsampleTileDataset(Dataset):
    """
    Creates matched LR/HR pairs by date, normalises to [0,1], downsamples to target
    resolution, and yields joint random crops of size (tile_size x tile_size).
    """
    def __init__(self, lr_dir, hr_dir, resolution_meters, tile_size=512, num_tiles=1000):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.resolution_meters = resolution_meters
        self.tile_size = tile_size
        self.num_tiles = num_tiles
        self.ghi_max = 1200.0
        from torchvision.transforms import RandomCrop
        self.cropper = RandomCrop(tile_size)

        self.lr_files = sorted(self.lr_dir.glob("*.tif"))
        self.hr_files = sorted(self.hr_dir.glob("*.tif"))

        lr_dict = {extract_first_date(f.name): f for f in self.lr_files if extract_first_date(f.name)}
        hr_dict = {extract_first_date(f.name): f for f in self.hr_files if extract_first_date(f.name)}
        common = sorted(set(lr_dict.keys()) & set(hr_dict.keys()))
        if not common:
            raise ValueError("No matching LR and HR pairs found based on acquisition date.")
        self.pairs = [(lr_dict[d], hr_dict[d]) for d in common]

    def _downsample(self, array, target_size):
        img = Image.fromarray(array)
        return np.array(img.resize(target_size, resample=Image.BICUBIC)).astype(np.float32)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):
        from torchvision.transforms import functional as TF

        lr_path, hr_path = self.pairs[idx % len(self.pairs)]
        with rasterio.open(lr_path) as src:
            lr = src.read(1).astype(np.float32)
        with rasterio.open(hr_path) as src:
            hr = src.read(1).astype(np.float32)

        # NaNs → 0 and clip
        lr = np.clip(np.nan_to_num(lr, nan=0.0), 0.0, self.ghi_max)
        hr = np.clip(np.nan_to_num(hr, nan=0.0), 0.0, self.ghi_max)

        # Bring to target resolution (assumes native HR ≈ 10 m/px)
        factor = self.resolution_meters / 10.0
        new_size = (int(lr.shape[1] / factor), int(lr.shape[0] / factor))
        lr = self._downsample(lr, new_size)
        hr = self._downsample(hr, new_size)

        # Normalise to [0,1]
        lr /= self.ghi_max
        hr /= self.ghi_max

        lr_t = torch.from_numpy(lr).unsqueeze(0)  # [1,H,W]
        hr_t = torch.from_numpy(hr).unsqueeze(0)

        # Joint random crop
        i, j, h, w = self.cropper.get_params(lr_t, (self.tile_size, self.tile_size))
        lr_crop = TF.crop(lr_t, i, j, h, w)
        hr_crop = TF.crop(hr_t, i, j, h, w)

        return lr_crop, hr_crop

# =======================
# Model: SegFormer b0 → regression head
# =======================
class SegFormerRegressor(nn.Module):
    def __init__(self, in_channels=1, weights="none"):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel

        if weights == "none":
            cfg = SegformerConfig(
                num_channels=in_channels,
                num_encoder_blocks=4,
                depths=[2,2,2,2],
                hidden_sizes=[32,64,160,256],
                patch_sizes=[7,3,3,3],
                strides=[4,2,2,2],
                num_attention_heads=[1,2,5,8],
                decoder_hidden_size=256
            )
            self.segformer = SegformerModel(cfg)
        elif weights == "imagenet":
            # Try local cache first, then fall back to Hugging Face hub
            try:
                self.segformer = SegformerModel.from_pretrained(
                    "/path_to_local_mit_b0", local_files_only=True
                )
            except Exception:
                self.segformer = SegformerModel.from_pretrained(
                    "nvidia/mit-b0", local_files_only=False
                )
            # Adapt 3→in_channels conv by averaging across RGB channels
            old = self.segformer.encoder.patch_embeddings[0].proj
            new = nn.Conv2d(in_channels, old.out_channels,
                            kernel_size=old.kernel_size, stride=old.stride,
                            padding=old.padding, bias=(old.bias is not None))
            with torch.no_grad():
                new.weight[:] = old.weight.mean(dim=1, keepdim=True)
                if old.bias is not None: new.bias[:] = old.bias
            self.segformer.encoder.patch_embeddings[0].proj = new
        else:
            raise ValueError("weights must be 'none' or 'imagenet'")

        self.reg_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = self.segformer(x).last_hidden_state   # [B, 256, H/4, W/4]
        y = self.reg_head(h)                      # [B, 1, H/4, W/4]
        y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return torch.clamp(y, 0.0, 1.0)

# =======================
# Train / Validate / Test
# =======================
def _first_existing(*cands):
    for c in cands:
        if c and Path(c).exists():
            return c
    return cands[0] if cands else None

def _resolve_train_dirs(base_lr_path, base_hr_path, train_subset_root):
    """
    Resolve training LR/HR directories:
      - If train_subset_root provided and contains 'lr' and 'hr' subfolders, use those.
      - If it points to LR or HR subset directly, infer the counterpart folder.
      - Else, default to base/train/lr and base/train/hr.
    """
    if train_subset_root:
        p = Path(train_subset_root)
        if (p / "lr").exists() and (p / "hr").exists():
            return str(p / "lr"), str(p / "hr")
        s = str(p)
        if "/model_input/lr/" in s:
            return s, s.replace("/model_input/lr/", "/model_input/hr/")
        if "/model_input/hr/" in s:
            return s.replace("/model_input/hr/", "/model_input/lr/"), s
        # Fallback: treat as explicit LR subset and try to infer HR by sibling "hr"
        if (p.parent / "hr").exists():
            return str(p), str(p.parent / "hr")
        raise ValueError("train_subset_root must have lr/hr subdirs OR be under model_input/{lr|hr}/...")
    # default
    return f"{base_lr_path}/train/lr", f"{base_hr_path}/train/hr"

def train_val_test_model(
    base_lr_path,
    base_hr_path,
    resolution_meters=40,
    weights="imagenet",
    lr=1e-4,
    num_epochs=30,
    batch_size=8,
    tile_size=512,
    train_tiles=1000,
    val_tiles=200,
    test_tiles=200,
    num_workers=4,
    train_subset_root=None,
    val_lr_dir=None, val_hr_dir=None, test_lr_dir=None, test_hr_dir=None,
    out_root="./models/segformer_b0",
    loss_ylim=None,
):
    # Device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Resolve split directories with fallbacks
    train_lr_dir, train_hr_dir = _resolve_train_dirs(base_lr_path, base_hr_path, train_subset_root)
    val_lr_dir  = val_lr_dir  or _first_existing(f"{base_lr_path}/val/lr",  f"{base_lr_path}/val_all")
    val_hr_dir  = val_hr_dir  or _first_existing(f"{base_hr_path}/val/hr",  f"{base_hr_path}/val_all")
    test_lr_dir = test_lr_dir or _first_existing(f"{base_lr_path}/test/lr", f"{base_lr_path}/test_all")
    test_hr_dir = test_hr_dir or _first_existing(f"{base_hr_path}/test/hr", f"{base_hr_path}/test_all")
    if not (val_lr_dir and val_hr_dir and test_lr_dir and test_hr_dir):
        raise FileNotFoundError("Validation/test directories not found (neither split nor *_all).")

    # Datasets / Loaders
    train_dataset = GHIDownsampleTileDataset(train_lr_dir, train_hr_dir,
                                             resolution_meters, tile_size, train_tiles)
    val_dataset   = GHIDownsampleTileDataset(val_lr_dir,   val_hr_dir,
                                             resolution_meters, tile_size, val_tiles)
    test_dataset  = GHIDownsampleTileDataset(test_lr_dir,  test_hr_dir,
                                             resolution_meters, tile_size, test_tiles)

    # Deterministic val/test order
    val_loader  = DataLoader(Subset(val_dataset,  list(range(len(val_dataset)))),  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(test_dataset, list(range(len(test_dataset)))), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)

    # Model
    model = SegFormerRegressor(in_channels=1, weights=weights).to(device)

    # Optionally freeze encoder for ImageNet start and train last stages + head
    if weights == "imagenet":
        for p in model.segformer.parameters(): p.requires_grad = False
        for p in model.segformer.encoder.block[2:].parameters(): p.requires_grad = True
        for p in model.reg_head.parameters(): p.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # Output dirs and run name
    stem = f"segformer_b0_{weights}_res{resolution_meters}m_tile{tile_size}_bs{batch_size}_lr{lr:.0e}_ep{num_epochs}"
    out_dir = Path(out_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{stem}_best.pth"
    last_path = out_dir / f"{stem}_last.pth"

    # Train/Val loop
    train_epoch_losses, val_losses = [], []
    t0_all = time.perf_counter()

    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        model.train()
        total = 0.0; n_batches = 0
        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(lr_batch)
                loss = criterion(pred, hr_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item(); n_batches += 1

        avg_train = total / max(1, n_batches)
        train_epoch_losses.append(avg_train)

        # Validation
        model.eval()
        vtot = 0.0; v_batches = 0
        with torch.no_grad(), torch.cuda.amp.autocast():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device, non_blocking=True)
                hr_batch = hr_batch.to(device, non_blocking=True)
                pred = model(lr_batch)
                vloss = criterion(pred, hr_batch)
                vtot += vloss.item(); v_batches += 1
        avg_val = vtot / max(1, v_batches)
        val_losses.append(avg_val)

        dt = time.perf_counter() - t0
        samples = n_batches * batch_size
        sps = samples / dt if dt > 0 else float("nan")
        print(f"Epoch {epoch+1}/{num_epochs} | Train {avg_train:.6f} | Val {avg_val:.6f} | "
              f"{dt:.1f}s | ~{sps:.1f} samples/s", flush=True)

        # Save last checkpoint each epoch (optionally choose best via early stopping)
        torch.save(model.state_dict(), last_path)

    # Loss plot
    plt.figure(figsize=(7,5))
    xs = range(1, len(train_epoch_losses)+1)
    plt.plot(xs, train_epoch_losses, label="Train", alpha=0.8)
    plt.plot(xs, val_losses,        label="Val",   alpha=0.8)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss curves")
    plt.xlim(1, len(train_epoch_losses))
    if loss_ylim is not None:
        plt.ylim(loss_ylim[0], loss_ylim[1])
    elif "loss_y_fixed" in EVAL_CFG and EVAL_CFG["loss_y_fixed"]:
        plt.ylim(EVAL_CFG.get("loss_y_min", 0.0), EVAL_CFG.get("loss_y_max", 0.10))
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "loss.pdf", bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to: {out_dir/'loss.png'}", flush=True)

    # If no specific "best" was tracked, reuse last
    if not best_path.exists():
        if last_path.exists():
            shutil.copy(last_path, best_path)
        else:
            torch.save(model.state_dict(), best_path)

    # ===== Test =====
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    preds_list, targets_list = [], []
    with torch.inference_mode():
        for lr_batch, hr_batch in test_loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            out = model(lr_batch).detach().cpu().numpy()
            tgt = hr_batch.numpy()
            preds_list.append(out); targets_list.append(tgt)

    preds_all   = np.concatenate(preds_list, axis=0).squeeze()
    targets_all = np.concatenate(targets_list, axis=0).squeeze()

    # Denormalise to physical units if configured
    if EVAL_CFG["eval_in_physical_units"]:
        s = EVAL_CFG["ghi_scale_max"]
        preds_all   *= s
        targets_all *= s

    # Optional unit conversion Wh/m^2 → W/m^2
    unit_in  = EVAL_CFG.get("unit_in",  "W/m^2")
    unit_out = EVAL_CFG.get("unit_out", unit_in)
    if unit_in == "Wh/m^2" and unit_out == "W/m^2":
        dt_minutes = float(EVAL_CFG.get("dt_minutes", 1.0))
        if dt_minutes <= 0:
            raise ValueError("EVAL_CFG['dt_minutes'] must be > 0 for unit conversion.")
        factor = 60.0 / dt_minutes
        preds_all   *= factor
        targets_all *= factor
    plot_unit = unit_out

    # === Scalar metrics
    metrics_eval = {
        "MSE": mse(preds_all, targets_all),
        "RMSE": float(np.sqrt(mse(preds_all, targets_all))),
        "MAE": mae(preds_all, targets_all),
        "MBE": mbe(preds_all, targets_all),
        "Correlation": corr_pearson(preds_all, targets_all),
    }

    # === PSD & band correlations
    f_true, Pxx_true, Pxx_pred, bin_corr = compute_psd_metrics(
        targets_all, preds_all, fs=EVAL_CFG["psd_fs"], bins_cfg=EVAL_CFG["psd_bins"]
    )
    metrics_eval.update({
        "PSD_low":     bin_corr[0],
        "PSD_lowmed":  bin_corr[1],
        "PSD_medhigh": bin_corr[2],
        "PSD_high":    bin_corr[3],
    })

    # === Plots
    # Histogram
    hr = EVAL_CFG["hist_range_w"] if plot_unit == "W/m^2" else EVAL_CFG["hist_range_wh"]
    plt.figure(figsize=(7,5))
    plt.hist(targets_all.ravel(), bins=EVAL_CFG["hist_bins"], range=hr,
             alpha=0.6, label="Ground truth", density=True)
    plt.hist(preds_all.ravel(),   bins=EVAL_CFG["hist_bins"], range=hr,
             alpha=0.6, label="Prediction",  density=True)
    plt.xlabel(f"GHI [{plot_unit}]"); plt.ylabel("Density")
    plt.title("GHI distribution"); plt.xlim(*hr); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "histogram.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "histogram.pdf", bbox_inches="tight")
    plt.close()

    # 2D density (hexbin) + classic scatter (square axes)
    sr = EVAL_CFG["scatter_range_w"] if plot_unit == "W/m^2" else EVAL_CFG["scatter_range_wh"]
    lo, hi = sr
    plt.figure(figsize=(6, 6))
    hb = plt.hexbin(preds_all.ravel(), targets_all.ravel(),
                    gridsize=EVAL_CFG["hexbin_gridsize"], bins='log',
                    extent=(lo, hi, lo, hi), mincnt=1)
    cb = plt.colorbar(hb); cb.set_label('log(count)')
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f"Prediction [{plot_unit}]"); plt.ylabel(f"Ground truth [{plot_unit}]")
    plt.title("Prediction vs Ground truth (hexbin)")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "density_scatter.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "density_scatter.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot([lo, hi], [lo, hi], lw=1.2)
    plt.scatter(preds_all.ravel(), targets_all.ravel(), s=2, alpha=0.25)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f"Prediction [{plot_unit}]"); plt.ylabel(f"Ground truth [{plot_unit}]")
    plt.title("Prediction vs Ground truth (scatter)")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "scatter.pdf", bbox_inches="tight")
    plt.close()

    # PSD
    eps = 1e-15
    plt.figure(figsize=(7,5))
    plt.semilogy(f_true, Pxx_true + eps, label="Ground Truth")
    plt.semilogy(f_true, Pxx_pred + eps, label="Prediction")
    plt.xlabel("Spatial frequency"); plt.ylabel("Power spectral density")
    plt.title("PSD comparison")
    plt.xlim(0, f_true.max())
    if EVAL_CFG.get("psd_y_fixed", False):
        plt.ylim(EVAL_CFG["psd_y_min"], EVAL_CFG["psd_y_max"])
    else:
        both = np.concatenate([Pxx_true, Pxx_pred])
        p1, p99 = EVAL_CFG["psd_y_percentile_clip"]
        lo_y = max(np.percentile(both, p1), 1e-15)
        hi_y = np.percentile(both, p99)
        if not (np.isfinite(lo_y) and np.isfinite(hi_y)) or hi_y <= lo_y:
            lo_y, hi_y = 1e-12, 1.0
        plt.ylim(lo_y, hi_y)
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "psd.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "psd.pdf", bbox_inches="tight")
    plt.close()

    # Save arrays + metrics + config
    np.save(out_dir / "psd_f.npy",     f_true)
    np.save(out_dir / "psd_true.npy",  Pxx_true)
    np.save(out_dir / "psd_pred.npy",  Pxx_pred)

    with open(out_dir / "evaluation_metrics.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["Metric", "Value"])
        for k, v in metrics_eval.items():
            w.writerow([k, v])

    with open(out_dir / "eval_config.json", "w") as fjs:
        json.dump(EVAL_CFG, fjs, indent=2)

    total_secs = time.perf_counter() - t0_all
    print("Eval metrics:", metrics_eval, flush=True)
    print(f"Artifacts saved to: {out_dir}", flush=True)
    print(f"Total runtime: {total_secs:.1f}s", flush=True)

    return {
        "final_train_loss": float(train_epoch_losses[-1]) if train_epoch_losses else float("nan"),
        "final_val_loss": float(val_losses[-1]) if val_losses else float("nan"),
        **metrics_eval,
        "out_dir": str(out_dir),
        "model_path": str(best_path),
        "total_runtime_s": float(total_secs),
    }

# =======================
# Example run(s)
# =======================
if __name__ == "__main__":
    # Main uses **DEFAULTS only. Edit paths/hyperparameters in DEFAULTS above.
    res = train_val_test_model(
        base_lr_path=DEFAULTS["base_lr_path"],
        base_hr_path=DEFAULTS["base_hr_path"],
        resolution_meters=DEFAULTS["resolution_meters"],
        weights=DEFAULTS["weights"],
        lr=DEFAULTS["lr"],
        num_epochs=DEFAULTS["num_epochs"],
        batch_size=DEFAULTS["batch_size"],
        tile_size=DEFAULTS["tile_size"],
        train_tiles=DEFAULTS["train_tiles"],
        val_tiles=DEFAULTS["val_tiles"],
        test_tiles=DEFAULTS["test_tiles"],
        num_workers=DEFAULTS["num_workers"],
        train_subset_root=DEFAULTS["train_subset_root"],
        val_lr_dir=DEFAULTS["val_lr_dir"],
        val_hr_dir=DEFAULTS["val_hr_dir"],
        test_lr_dir=DEFAULTS["test_lr_dir"],
        test_hr_dir=DEFAULTS["test_hr_dir"],
        out_root=DEFAULTS["out_root"],
        loss_ylim=DEFAULTS["loss_ylim"],
    )
    try:
        print(json.dumps(res, indent=2))
    except Exception:
        print(res)
