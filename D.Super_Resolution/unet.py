#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net Super-Resolution (training + evaluation)

What this script does
- Trains a U-Net (segmentation_models_pytorch) to super-resolve low-resolution GHI tiles into high-resolution targets.
- Pairs LR↔HR scenes by (site, date), samples random crops per epoch, and evaluates on val/test.
- Saves model checkpoint, figures (loss / histogram / scatter / PSD), arrays, and a central CSV log.

Inputs on disk (fill in with your own paths)
- base_lr_path/
    ├── train/lr/*.tif
    ├── val/lr/*.tif      (fallback: val_all/*.tif)
    └── test/lr/*.tif     (fallback: test_all/*.tif)
- base_hr_path/
    ├── train/hr/*.tif
    ├── val/hr/*.tif      (fallback: val_all/*.tif)
    └── test/hr/*.tif     (fallback: test_all/*.tif)

Filename conventions for pairing
- LR:  <site>__low_res_ghi_YYYYMMDD_*.tif   (e.g., cabauw__low_res_ghi_20230415_xx.tif)
- HR:  heleo_ghi_S2A_MSIL1C_YYYYMMDDT..._T31UFT_....tif  (tile → site via mapping below)

Outputs
- Run folder under out_root with:
  • model checkpoint (.pth)
  • loss.png
  • histogram.(png|pdf), scatter.(png|pdf), density_scatter.(png|pdf)
  • psd.(png|pdf) and corresponding .npy arrays
  • evaluation_metrics.csv
  • eval_config.json
- A central CSV append log at results_csv

Requirements
- Python 3.9+
- pip install: torch, segmentation_models_pytorch, numpy, rasterio, pillow, matplotlib, scipy, torchvision

Usage (quick start)
-------------------
# 1) Set placeholders in DEFAULTS below (paths, hyperparameters)
# 2) Run this script directly:  python model_unet_v2.py
# The __main__ block calls train_val_test_model(**DEFAULTS).
"""

# =======================
# Defaults (edit these)
# =======================
DEFAULTS = {
    "base_lr_path": "/path_to_lr_root",
    "base_hr_path": "/path_to_hr_root",

    # Change per subset; set to None to disable
    "train_subset_root": "/path_to_train_subset",  # or None

    "resolution_meters": 40,
    "encoder": "mobilenet_v2",
    "encoder_weight": "imagenet",
    "lr": 1e-4,
    "num_epochs": 15,
    "batch_size": 8,
    "tile_size": 512,

    # ≈30 tiles per scene for ~30 scenes
    "train_tiles": 9030,

    # keep val/test > 0; 400 is fine
    "val_tiles": 400,
    "test_tiles": 400,

    "out_root": "/path_to_output_models/unet",
    "loss_ylim": None,  # or (ymin, ymax)

    # Budget label (number of scenes) in naming; autodetect if None
    "budget": None,

    # Fixed paths for val/test (or None to use *_all fallbacks)
    "val_lr_dir": None,
    "val_hr_dir": None,
    "test_lr_dir": None,
    "test_hr_dir": None,

    # Central CSV log (append)
    "results_csv": "/path_to_results_dir/results.csv",
}

# =======================
# Created on Thu Aug 28 14:42:14 2025
# Author: Lot
# =======================

import os, re, time, csv, json
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF

import segmentation_models_pytorch as smp

import matplotlib
matplotlib.use("Agg")  # required on headless systems (e.g., DelftBlue)
import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# =======================
# Config for consistent evaluation/plots
# =======================
EVAL_CFG = {
    # If you normalise to [0,1] during training: set True and provide physical scale
    "eval_in_physical_units": True,
    "ghi_scale_max": 20.0,   # Wh/m^2 (clipped/scaled here)
    "unit_in": "Wh/m^2",
    "unit_out": "W/m^2",
    "dt_minutes": 1.0,

    # Loss-plot axes
    "loss_y_fixed": True,
    "loss_y_min": 0.0,
    "loss_y_max": 0.10,

    # Histogram settings
    "hist_bins": 120,
    "hist_range_wh": (0.0, 20.0),
    "hist_range_w": (0.0, 1200.0),

    # Scatter/hexbin settings
    "hexbin_gridsize": 120,
    "scatter_range_wh": (0.0, 20.0),
    "scatter_range_w": (0.0, 1200.0),

    # PSD settings
    "psd_fs": 1.0,  # example sampling (arbitrary in this 1D series)
    "psd_y_fixed": False,
    "psd_y_min": 1e-10,
    "psd_y_max": 1e0,
    "psd_y_percentile_clip": (1, 99),
    # Frequency bands (low → high); None = Nyquist
    "psd_bins": [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, None)],
}

SITE_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+)__")  # match 'cabauw__' etc.

# ===== Tile → Site mapping =====
TILE_TO_SITE = {
    "T31UFT": "cabauw",
    "T31UDP": "sirta",
    "T30SVG": "granada",
    "T35VLJ": "hyytiala",
}

# --- Parsers ---
_site_date_lr = re.compile(r'^(?P<site>[a-z]+)__low_res_ghi_(?P<date>\d{8})', re.IGNORECASE)
_date_hr      = re.compile(r'(?P<date>\d{8})T')     # YYYYMMDD before the T
_tile_hr      = re.compile(r'(T\d{2}[A-Z]{3})')     # e.g. T31UFT, T30SVG, ...


def lr_key(p):
    """cabauw__low_res_ghi_20230415_nn.tif → ('cabauw','20230415')"""
    m = _site_date_lr.match(p.name)
    if not m:
        return None
    site = m.group('site').lower()
    date = m.group('date')
    return site, date


def hr_key(p):
    """
    heleo_ghi_S2A_MSIL1C_20230415T104621_..._T31UFT_....tif
        → tile 'T31UFT' → 'cabauw', date '20230415'
    """
    md = _date_hr.search(p.name)
    mt = _tile_hr.search(p.name)
    if not (md and mt):
        return None
    date = md.group('date')
    tile = mt.group(1).upper()
    site = TILE_TO_SITE.get(tile)
    if not site:
        return None
    return site, date


def build_lr_hr_pairs(lr_dir, hr_dir):
    """Create matched (LR, HR) pairs based on (site, date)."""
    lr_dir, hr_dir = Path(lr_dir), Path(hr_dir)
    lr_files = sorted(lr_dir.glob("*.tif*"))
    hr_files = sorted(hr_dir.glob("*.tif*"))

    lr_map = {}
    for p in lr_files:
        k = lr_key(p)
        if k:
            lr_map[k] = min(p, lr_map.get(k, p), key=lambda q: q.name)

    hr_map = {}
    for p in hr_files:
        k = hr_key(p)
        if k:
            hr_map[k] = min(p, hr_map.get(k, p), key=lambda q: q.name)

    common = sorted(set(lr_map.keys()) & set(hr_map.keys()))
    pairs = [(lr_map[k], hr_map[k]) for k in common]

    # Debug prints (optional)
    print(f"[MATCH] LR files parsed: {len(lr_map)} | HR files parsed: {len(hr_map)}")
    print(f"[MATCH] Intersect (site,date): {len(common)}")
    missing_in_hr = sorted(set(lr_map.keys()) - set(hr_map.keys()))
    missing_in_lr = sorted(set(hr_map.keys()) - set(lr_map.keys()))
    if missing_in_hr[:5]:
        print("[WARN] first missing in HR:", missing_in_hr[:5])
    if missing_in_lr[:5]:
        print("[WARN] first missing in LR:", missing_in_lr[:5])

    return pairs


def _extract_site_and_date(filename):
    msite = SITE_PREFIX_RE.match(filename)
    site = msite.group(1) if msite else None
    date = extract_first_date(filename)
    return site, date


def extract_first_date(filename):
    matches = re.findall(r"20\d{6}", filename)
    return matches[0] if matches else None


class GHIDownsampleTileDataset(Dataset):
    """
    Yields random tiles from aligned LR/HR pairs at the same target resolution (normalised to [0,1]).
    """
    def __init__(self, lr_dir, hr_dir, resolution_meters, tile_size=512, num_tiles=1000):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.resolution_meters = resolution_meters
        self.tile_size = tile_size
        self.num_tiles = num_tiles

        self.lr_files = sorted(self.lr_dir.glob("*.tif"))
        self.hr_files = sorted(self.hr_dir.glob("*.tif"))

        self.cropper = RandomCrop(tile_size)
        self.ghi_max = 20.0  # clip/scale

        # Pair LR/HR on (site, date)
        lr_map = {}
        for f in self.lr_files:
            k = lr_key(f)
            if k:
                lr_map[k] = min(f, lr_map.get(k, f), key=lambda p: p.name)

        hr_map = {}
        for f in self.hr_files:
            k = hr_key(f)
            if k:
                hr_map[k] = min(f, hr_map.get(k, f), key=lambda p: p.name)

        common_keys = sorted(set(lr_map) & set(hr_map))
        self.pairs = [(lr_map[k], hr_map[k]) for k in common_keys]

        if not self.pairs:
            raise ValueError(
                f"No matching LR and HR pairs found on (site,date) in:\n{lr_dir}\n{hr_dir}"
            )

    def _resize(self, array, target_size):
        image = Image.fromarray(array)
        resized = image.resize(target_size, resample=Image.BICUBIC)
        return np.array(resized).astype(np.float32)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):
        # Cycle through pairs; variation via random crop
        file_idx = idx % len(self.pairs)
        lr_path, hr_path = self.pairs[file_idx]

        with rasterio.open(lr_path) as src:
            lr = src.read(1).astype(np.float32)
        with rasterio.open(hr_path) as src:
            hr = src.read(1).astype(np.float32)

        # Clip & NaN→0
        lr = np.clip(np.nan_to_num(lr, nan=0.0), 0, self.ghi_max)
        hr = np.clip(np.nan_to_num(hr, nan=0.0), 0, self.ghi_max)

        # Bring to target resolution (assumption: HR native 10 m/pixel)
        factor = self.resolution_meters / 10.0
        new_size = (int(lr.shape[1] / factor), int(lr.shape[0] / factor))
        lr = self._resize(lr, new_size)
        hr = self._resize(hr, new_size)

        # Normalise
        lr_norm = lr / self.ghi_max
        hr_norm = hr / self.ghi_max

        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_norm).unsqueeze(0)

        # Joint random crop
        i, j, h, w = self.cropper.get_params(lr_tensor, (self.tile_size, self.tile_size))
        lr_crop = TF.crop(lr_tensor, i, j, h, w)
        hr_crop = TF.crop(hr_tensor, i, j, h, w)

        return lr_crop, hr_crop


class ReLUWrappedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        return F.relu(self.base(x))


def _mask_valid(y, t):
    y = np.asarray(y).ravel()
    t = np.asarray(t).ravel()
    m = np.isfinite(y) & np.isfinite(t)
    return y[m], t[m]

def mae(y, t): y,t = _mask_valid(y,t); return float(np.mean(np.abs(y - t)))
def mbe(y, t): y,t = _mask_valid(y,t); return float(np.mean(y - t))
def mse(y, t): y,t = _mask_valid(y,t); return float(np.mean((y - t)**2))
def corr_pearson(y, t):
    y,t = _mask_valid(y,t)
    if y.size < 2: return np.nan
    try:
        r,_ = pearsonr(y, t)
    except Exception:
        r = np.nan
    return float(r)

def welch_psd(arr, fs=1.0, nperseg=None):
    arr = np.asarray(arr).ravel()
    arr = arr - np.nanmean(arr)
    if nperseg is None:
        nperseg = min(len(arr), max(256, len(arr)//4))
    f, Pxx = welch(arr, fs=fs, nperseg=nperseg)
    return f, Pxx

def integrate_psd_bands(f, Pxx, bins_cfg):
    bands = []
    for low, high in bins_cfg:
        hi = f.max() if high is None else high
        mask = (f >= low) & (f < hi)
        if np.any(mask):
            bands.append(float(np.trapz(Pxx[mask], f[mask])))
        else:
            bands.append(np.nan)
    return bands


# =======================
# Train + Eval
# =======================
def train_val_test_model(
    base_lr_path, base_hr_path, resolution_meters,
    encoder="resnet34", encoder_weight=None,
    lr=1e-4, num_epochs=30, batch_size=2, tile_size=512,
    train_tiles=1000, val_tiles=200, test_tiles=200,
    out_root="/path_to_output_models/unet",
    loss_ylim=None,
    # Only the TRAIN subset folder for learning-curve budgets
    train_subset_root=None,
    # Budget (number of scenes) in name; autodetect if None
    budget=None,
    # Fixed paths for val/test
    val_lr_dir=None, val_hr_dir=None, test_lr_dir=None, test_hr_dir=None,
    # Central CSV log
    results_csv="/path_to_results_dir/results.csv",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== Data dirs ====
    def _first_existing(*cands):
        from pathlib import Path as _P
        for c in cands:
            if c and _P(c).exists():
                return c
        return cands[0]

    if train_subset_root is not None:
        p = Path(train_subset_root)
        # 1) Old pattern: subset_root/lr and subset_root/hr exist
        if (p / "lr").exists() and (p / "hr").exists():
            train_lr_dir = str(p / "lr")
            train_hr_dir = str(p / "hr")
        else:
            # 2) New: subset_root points directly to an LR or HR dir
            s = str(p)
            if "/model_input/lr/" in s:
                train_lr_dir = s
                train_hr_dir = s.replace("/model_input/lr/", "/model_input/hr/")
            elif "/model_input/hr/" in s:
                train_hr_dir = s
                train_lr_dir = s.replace("/model_input/hr/", "/model_input/lr/")
            else:
                raise ValueError(
                    "train_subset_root must have lr/hr subdirs OR be a subset dir under model_input/{lr|hr}/..."
                )
        if budget is None:
            budget = len(list(Path(train_lr_dir).glob("*.tif*")))
    else:
        train_lr_dir = f"{base_lr_path}/train/lr"
        train_hr_dir = f"{base_hr_path}/train/hr"
        if budget is None:
            budget = len(list(Path(train_lr_dir).glob('*.tif*')))

    # Defaults for val/test with fallback to *_all
    val_lr_dir  = val_lr_dir  or _first_existing(f"{base_lr_path}/val/lr",  f"{base_lr_path}/val_all")
    val_hr_dir  = val_hr_dir  or _first_existing(f"{base_hr_path}/val/hr",  f"{base_hr_path}/val_all")
    test_lr_dir = test_lr_dir or _first_existing(f"{base_lr_path}/test/lr", f"{base_lr_path}/test_all")
    test_hr_dir = test_hr_dir or _first_existing(f"{base_hr_path}/test/hr", f"{base_hr_path}/test_all")

    # ==== Datasets ====
    train_dataset = GHIDownsampleTileDataset(train_lr_dir, train_hr_dir,
                                             resolution_meters, tile_size, train_tiles)
    val_dataset   = GHIDownsampleTileDataset(val_lr_dir,   val_hr_dir,
                                             resolution_meters, tile_size, val_tiles)
    test_dataset  = GHIDownsampleTileDataset(test_lr_dir,  test_hr_dir,
                                             resolution_meters, tile_size, test_tiles)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # ==== Model ====
    unet_base = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weight, in_channels=1, classes=1)
    model = ReLUWrappedModel(unet_base).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_epoch_losses, val_losses = [], []
    epoch_times = []

    for epoch in range(num_epochs):
        model.train()
        t0 = time.perf_counter()
        total_train = 0.0

        for lr_batch, hr_batch in train_loader:
            lr_batch, hr_batch = lr_batch.to(device, non_blocking=True), hr_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(lr_batch)
            loss = criterion(out, hr_batch)
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / max(1, len(train_loader))
        train_epoch_losses.append(avg_train)

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch, hr_batch = lr_batch.to(device, non_blocking=True), hr_batch.to(device, non_blocking=True)
                out = model(lr_batch)
                loss = criterion(out, hr_batch)
                total_val += loss.item()
        avg_val = total_val / max(1, len(val_loader))
        val_losses.append(avg_val)

        ep_time = time.perf_counter() - t0
        epoch_times.append(ep_time)
        print(f"Epoch {epoch+1}/{num_epochs} | Train {avg_train:.6f} | Val {avg_val:.6f} | {ep_time:.2f}s")

    # ==== Naming (incl. number of scenes + tiles) ====
    ew = encoder_weight if encoder_weight is not None else "none"
    model_stem = (
        f"unet_{encoder}_{ew}"
        f"_res{resolution_meters}m_tile{tile_size}"
        f"_bs{batch_size}_lr{lr:.0e}_ep{num_epochs}"
        f"_sc{budget}_trtiles{train_tiles}_val{val_tiles}_test{test_tiles}"
    )
    out_dir = Path(out_root) / model_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_stem}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # ==== Loss plot (consistent axes) ====
    plt.figure(figsize=(6, 4))
    xs = list(range(1, num_epochs + 1))
    plt.plot(xs, train_epoch_losses, label="Train Loss", alpha=0.8)
    plt.plot(xs, val_losses,        label="Val Loss",   alpha=0.8)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.xlim(1, num_epochs)
    if loss_ylim is not None:
        plt.ylim(loss_ylim[0], loss_ylim[1])
    elif EVAL_CFG["loss_y_fixed"]:
        plt.ylim(EVAL_CFG["loss_y_min"], EVAL_CFG["loss_y_max"])
    else:
        ymax = max(max(train_epoch_losses), max(val_losses))
        plt.ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{model_stem}_loss.png", dpi=150)
    plt.close()
    print(f"Loss plot saved to: {out_dir / f'{model_stem}_loss.png'}")

    # ==== Test + collect arrays (also store LR input) ====
    model.eval()
    total_test_loss = 0.0
    preds_list, targets_list, inputs_lr_list = [], [], []
    with torch.no_grad():
        for lr_batch, hr_batch in test_loader:
            # Store LR input (normalised) before model
            inputs_lr_list.append(lr_batch.cpu().numpy())
            lr_batch = lr_batch.to(device, non_blocking=True)
            out = model(lr_batch).detach().cpu().numpy()
            tgt = hr_batch.numpy()
            preds_list.append(out)
            targets_list.append(tgt)
            # test MSE over this batch
            total_test_loss += np.mean((out - tgt)**2)
    avg_test_loss = total_test_loss / max(1, len(test_loader))
    print(f"FINAL TEST LOSS = {avg_test_loss:.6f}")

    preds_all   = np.concatenate(preds_list, axis=0).squeeze()
    targets_all = np.concatenate(targets_list, axis=0).squeeze()
    inputs_lr   = np.concatenate(inputs_lr_list, axis=0).squeeze()
    if EVAL_CFG["eval_in_physical_units"]:
        s = EVAL_CFG["ghi_scale_max"]
        preds_all   = preds_all * s
        targets_all = targets_all * s
        inputs_lr   = inputs_lr * s

    # Convert Wh/m^2 → W/m^2 if requested
    unit_in  = EVAL_CFG.get("unit_in", "Wh/m^2")
    unit_out = EVAL_CFG.get("unit_out", "Wh/m^2")
    if unit_in == "Wh/m^2" and unit_out == "W/m^2":
        dt_h = EVAL_CFG.get("dt_minutes", 1.0) / 60.0  # 1 min -> 1/60 h
        factor = 1.0 / dt_h                            # = 60 for 1-minute
        preds_all   = preds_all * factor
        targets_all = targets_all * factor
        inputs_lr   = inputs_lr * factor
    plot_unit = unit_out

    # === Scalar metrics
    metrics_eval = {
        "MSE": mse(preds_all, targets_all),
        "RMSE": float(np.sqrt(mse(preds_all, targets_all))),
        "MAE": mae(preds_all, targets_all),
        "MBE": mbe(preds_all, targets_all),
        "Correlation": corr_pearson(preds_all, targets_all),
    }

    # === Residuals
    res_pred = preds_all - targets_all
    res_inp  = inputs_lr  - targets_all

    # === PSD (GT, Pred, Input, Residuals) on a shared frequency grid
    def shared_welch(arr_list, fs=1.0):
        Lmin = min(np.asarray(a).size for a in arr_list)
        nperseg = min(Lmin, max(256, Lmin // 4))  # same rule as before, but shared
        out = []
        for a in arr_list:
            a = np.asarray(a).ravel()
            a = a - np.nanmean(a)
            f, Pxx = welch(a, fs=fs, nperseg=nperseg)
            out.append(Pxx)
        return f, out

    f_shared, [Pxx_true, Pxx_pred, Pxx_inp, Pxx_err_pred, Pxx_err_inp] = shared_welch(
        [targets_all, preds_all, inputs_lr, res_pred, res_inp],
        fs=EVAL_CFG["psd_fs"]
    )

    # === Band integrals on shared f
    bands_pred     = integrate_psd_bands(f_shared, Pxx_pred,     EVAL_CFG["psd_bins"])
    bands_err_pred = integrate_psd_bands(f_shared, Pxx_err_pred, EVAL_CFG["psd_bins"])
    bands_err_inp  = integrate_psd_bands(f_shared, Pxx_err_inp,  EVAL_CFG["psd_bins"])

    metrics_eval.update({
        "PSD_low":     bands_pred[0],
        "PSD_lowmed":  bands_pred[1],
        "PSD_medhigh": bands_pred[2],
        "PSD_high":    bands_pred[3],
        "PSD_low_err_pred":      bands_err_pred[0],
        "PSD_lowmed_err_pred":   bands_err_pred[1],
        "PSD_medhigh_err_pred":  bands_err_pred[2],
        "PSD_high_err_pred":     bands_err_pred[3],
        "PSD_low_err_input":     bands_err_inp[0],
        "PSD_lowmed_err_input":  bands_err_inp[1],
        "PSD_medhigh_err_input": bands_err_inp[2],
        "PSD_high_err_input":    bands_err_inp[3],
    })

    # === Plots
    # Histogram (distribution)
    hr = EVAL_CFG["hist_range_w"] if plot_unit == "W/m^2" else EVAL_CFG["hist_range_wh"]
    plt.figure(figsize=(7, 5))
    plt.hist(targets_all.ravel(), bins=EVAL_CFG["hist_bins"], range=hr,
             alpha=0.6, label="Ground truth", density=True)
    plt.hist(preds_all.ravel(),   bins=EVAL_CFG["hist_bins"], range=hr,
             alpha=0.6, label="Prediction",  density=True)
    plt.xlabel(f"GHI [{plot_unit}]"); plt.ylabel("Density")
    plt.title("GHI distribution")
    plt.xlim(*hr); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "histogram.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "histogram.pdf", bbox_inches="tight")
    plt.close()

    # 2D density (hexbin)
    sr = EVAL_CFG["scatter_range_w"] if plot_unit == "W/m^2" else EVAL_CFG["scatter_range_wh"]
    lo, hi = sr
    plt.figure(figsize=(6, 6))  # square canvas
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

    # Classic scatter (square axes)
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

    # PSD: GT vs Pred vs Low-res input (three curves) with shared f
    eps = 1e-15
    plt.figure(figsize=(7, 5))
    plt.semilogy(f_shared, Pxx_true + eps, label="Ground Truth")
    plt.semilogy(f_shared, Pxx_pred + eps, label="Prediction")
    plt.semilogy(f_shared, Pxx_inp  + eps, label="Low-res input")

    plt.xlabel("Frequency"); plt.ylabel("Power Spectral Density")
    plt.title("PSD Comparison (GT / Pred / LR input)")
    plt.xlim(0, f_shared.max())
    if EVAL_CFG["psd_y_fixed"]:
        plt.ylim(EVAL_CFG["psd_y_min"], EVAL_CFG["psd_y_max"])
    else:
        both = np.concatenate([Pxx_true, Pxx_pred, Pxx_inp])
        p1, p99 = EVAL_CFG["psd_y_percentile_clip"]
        lo_y = max(np.percentile(both, p1), 1e-15)
        hi_y = np.percentile(both, p99)
        if not np.isfinite(lo_y) or not np.isfinite(hi_y) or hi_y <= lo_y:
            lo_y, hi_y = 1e-12, 1.0
        plt.ylim(lo_y, hi_y)
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "psd.png", dpi=600, bbox_inches="tight")
    plt.savefig(out_dir / "psd.pdf", bbox_inches="tight")
    plt.close()

    # Save arrays with shared f
    np.save(out_dir / "psd_f.npy",            f_shared)
    np.save(out_dir / "psd_true.npy",         Pxx_true)
    np.save(out_dir / "psd_pred.npy",         Pxx_pred)
    np.save(out_dir / "psd_input_lr.npy",     Pxx_inp)
    np.save(out_dir / "psd_error_pred.npy",   Pxx_err_pred)
    np.save(out_dir / "psd_error_input.npy",  Pxx_err_inp)

    # ==== Metrics + config save ====
    with open(out_dir / "evaluation_metrics.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["Metric", "Value"])
        w.writerow(["Final train loss", float(train_epoch_losses[-1])])
        w.writerow(["Final val loss",   float(val_losses[-1])])
        w.writerow(["Final test loss",  float(avg_test_loss)])
        w.writerow(["MAE",              metrics_eval["MAE"]])
        w.writerow(["MBE",              metrics_eval["MBE"]])
        w.writerow(["MSE",              metrics_eval["MSE"]])
        w.writerow(["Correlation",      metrics_eval["Correlation"]])
        w.writerow(["PSD_low",          metrics_eval["PSD_low"]])
        w.writerow(["PSD_lowmed",       metrics_eval["PSD_lowmed"]])
        w.writerow(["PSD_medhigh",      metrics_eval["PSD_medhigh"]])
        w.writerow(["PSD_high",         metrics_eval["PSD_high"]])

        # Residual PSD band integrals
        w.writerow(["PSD_low_err_pred",      metrics_eval["PSD_low_err_pred"]])
        w.writerow(["PSD_lowmed_err_pred",   metrics_eval["PSD_lowmed_err_pred"]])
        w.writerow(["PSD_medhigh_err_pred",  metrics_eval["PSD_medhigh_err_pred"]])
        w.writerow(["PSD_high_err_pred",     metrics_eval["PSD_high_err_pred"]])
        w.writerow(["PSD_low_err_input",     metrics_eval["PSD_low_err_input"]])
        w.writerow(["PSD_lowmed_err_input",  metrics_eval["PSD_lowmed_err_input"]])
        w.writerow(["PSD_medhigh_err_input", metrics_eval["PSD_medhigh_err_input"]])
        w.writerow(["PSD_high_err_input",    metrics_eval["PSD_high_err_input"]])

    with open(out_dir / "eval_config.json", "w") as fjs:
        json.dump(EVAL_CFG, fjs, indent=2)

    # Central CSV (append)
    Path(DEFAULTS["results_csv"]).parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp","budget","model_type","encoder","encoder_weight",
        "tile_size","batch_size","lr","epochs",
        "train_tiles","val_tiles","test_tiles",
        "final_train_loss","final_val_loss","final_test_loss",
        "MAE","MBE","MSE","Correlation","model_path"
    ]
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"), budget, "unet", encoder, (encoder_weight or "None"),
        tile_size, batch_size, lr, num_epochs,
        train_tiles, val_tiles, test_tiles,
        float(train_epoch_losses[-1]), float(val_losses[-1]), float(avg_test_loss),
        metrics_eval["MAE"], metrics_eval["MBE"], metrics_eval["MSE"], metrics_eval["Correlation"],
        str(model_path)
    ]
    write_header = not os.path.exists(DEFAULTS["results_csv"])
    with open(DEFAULTS["results_csv"], "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)

    # Summary
    print("Eval metrics:", metrics_eval)
    print(f"Artifacts saved to: {out_dir}")

    return {
        "final_train_loss": float(train_epoch_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "final_test_loss": float(avg_test_loss),
        **metrics_eval,
        "out_dir": str(out_dir),
        "model_path": str(model_path),
    }


# =======================
# Example run(s)
# =======================
if __name__ == "__main__":
    # The main entry point uses **DEFAULTS only. Edit paths/hyperparameters above.
    res = train_val_test_model(
        base_lr_path=DEFAULTS["base_lr_path"],
        base_hr_path=DEFAULTS["base_hr_path"],
        resolution_meters=DEFAULTS["resolution_meters"],
        encoder=DEFAULTS["encoder"],
        encoder_weight=DEFAULTS["encoder_weight"],
        lr=DEFAULTS["lr"],
        num_epochs=DEFAULTS["num_epochs"],
        batch_size=DEFAULTS["batch_size"],
        tile_size=DEFAULTS["tile_size"],
        train_tiles=DEFAULTS["train_tiles"],
        val_tiles=DEFAULTS["val_tiles"],
        test_tiles=DEFAULTS["test_tiles"],
        out_root=DEFAULTS["out_root"],
        loss_ylim=DEFAULTS["loss_ylim"],
        train_subset_root=DEFAULTS["train_subset_root"],
        budget=DEFAULTS["budget"],
        val_lr_dir=DEFAULTS["val_lr_dir"],
        val_hr_dir=DEFAULTS["val_hr_dir"],
        test_lr_dir=DEFAULTS["test_lr_dir"],
        test_hr_dir=DEFAULTS["test_hr_dir"],
        results_csv=DEFAULTS["results_csv"],
    )
    print(json.dumps(res, indent=2))
