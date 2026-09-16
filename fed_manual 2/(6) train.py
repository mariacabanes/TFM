"""
train_all.py — Unified training pipeline
=========================================
Trains both models in sequence:

  1. UNet  (dental X-ray segmentation)
     Input  : outputs/split/train/images  +  annotations.json
     Output : outputs/models/<run>/best_unet.pth

  2. ImplantClassifier  (EfficientNet + GNN, brand + diameter)
     Input  : outputs/dataset_split/train/features_augmented.csv
              outputs/dataset_split/val/features.csv
     Output : outputs/models/EfficientNetGNN_best_*.pt

Usage:
    python train_all.py
"""

import os
import re
import gc
import math
import copy
import datetime
import numpy as np
import platform
from pathlib import Path
from typing import Optional
from collections import Counter

import blosc
import cv2
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast

# ── Model imports ────────────────────────────────────────────────────────────
from models.Unet import UNet as SegModel
from models.ImplantClassifier import ImplantClassifier
# ─────────────────────────────────────────────────────────────────────────────

import json as _json

# =============================================================================
#  SHARED PATHS
# =============================================================================

_here         = Path(__file__).parent.resolve()
OUTPUT_DIR    = _here / "outputs"
ANALYSIS_PATH = OUTPUT_DIR / "analysis_results.json"

# UNet paths
SPLIT_DIR         = OUTPUT_DIR / "dataset_split"
TRAIN_IMAGES      = str(SPLIT_DIR / "train" / "images")
TRAIN_ANNOTATIONS = str(SPLIT_DIR / "train" / "annotations.json")
VAL_IMAGES        = str(SPLIT_DIR / "val"   / "images")
VAL_ANNOTATIONS   = str(SPLIT_DIR / "val"   / "annotations.json")

# ImplantClassifier paths
DATASET_SPLIT_DIR = OUTPUT_DIR / "dataset_split"
TRAIN_DIR         = DATASET_SPLIT_DIR / "train"
VAL_DIR           = DATASET_SPLIT_DIR / "val"

# Shared models dir (timestamped per run)
_run_name  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = OUTPUT_DIR / "models" / _run_name
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
#  HELPERS
# =============================================================================

def _bytes_to_gb(b: int) -> float:
    return b / 1024 ** 3

def _section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def export_model(name, mdl):
    fp = MODELS_DIR / f"{name}.model"
    with open(fp, 'wb') as fh:
        fh.write(dill.dumps(mdl))

def export_model_grads(name, data, grads=True):
    ext = '.grads' if grads else '.weights'
    fp  = MODELS_DIR / f"{name}{ext}"
    with open(fp, 'wb') as fh:
        fh.write(blosc.compress(dill.dumps(data), cname='zstd',
                                shuffle=blosc.BITSHUFFLE, clevel=9))
    print(f"Saved: {fp}")


# ═════════════════════════════════════════════════════════════════════════════
#
#   PART 1 — UNet  (dental X-ray segmentation)
#
# ═════════════════════════════════════════════════════════════════════════════

def train_unet():
    _section("PART 1 — UNet Training  (dental X-ray segmentation)")

    # ── Sanity checks ────────────────────────────────────────────────
    if not ANALYSIS_PATH.exists():
        print(f"[X] analysis_results.json not found: {ANALYSIS_PATH}")
        print("    Run dataset_analysis.py first.")
        raise SystemExit(1)

    with open(ANALYSIS_PATH) as f:
        _analysis = _json.load(f)

    print(f"[INFO] Train      : {TRAIN_IMAGES}")
    print(f"[INFO] Val        : {VAL_IMAGES}")
    print(f"[INFO] Models dir : {MODELS_DIR}")
    print(f"[INFO] Device     : {DEVICE}")

    # ── Hyperparameters ──────────────────────────────────────────────
    EPOCHS        = 100
    LEARNING_RATE = 1e-3
    PATIENCE      = 15
    IMG_SIZE      = 384   # capped by native resolution below

    # ── COCO categories ──────────────────────────────────────────────
    train_coco = COCO(TRAIN_ANNOTATIONS)
    val_coco   = COCO(VAL_ANNOTATIONS)

    categories      = train_coco.loadCats(train_coco.getCatIds())
    cat_ids         = [cat['id'] for cat in categories]
    cat_id_to_index = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
    NUM_CLASSES     = len(categories) + 1

    print(f"\nCategories ({len(categories)}):")
    for cat in categories:
        print(f"  [{cat['id']}] {cat['name']}  ->  class index {cat_id_to_index[cat['id']]}")
    print(f"Total output classes (including background): {NUM_CLASSES}")

    # ── Load model ───────────────────────────────────────────────────
    model = SegModel(n_channels=3, n_classes=NUM_CLASSES, bilinear=True).to(DEVICE)
    #model.use_checkpointing()
    print(f"\nTotal parameters     : {sum(p.numel() for p in model.parameters()):,}")
    print("Gradient checkpointing enabled.")

    # ── Native resolution cap ────────────────────────────────────────
    def _native_resolution(analysis: dict) -> Optional[int]:
        try:
            import statistics
            ann_path = analysis.get("cleaned_annotations") or analysis.get("source_annotations")
            if not ann_path or not Path(ann_path).exists():
                return None
            with open(ann_path) as f:
                coco_data = _json.load(f)
            widths  = [img["width"]  for img in coco_data.get("images", []) if img.get("width",  0) > 0]
            heights = [img["height"] for img in coco_data.get("images", []) if img.get("height", 0) > 0]
            if not widths or not heights:
                return None
            native = int(min(statistics.median(widths), statistics.median(heights)))
            return max(32, (native // 32) * 32)
        except Exception:
            return None

    _native = _native_resolution(_analysis)
    if _native is not None and _native < IMG_SIZE:
        print(f"[INFO] Capping IMG_SIZE {IMG_SIZE} -> {_native}  (native ≈ {_native}px)")
        IMG_SIZE = _native
    else:
        print(f"[INFO] IMG_SIZE={IMG_SIZE}px  (native ≈ {_native}px)")

    # ── Auto-tune batch size ─────────────────────────────────────────
    def _free_vram():
        if not torch.cuda.is_available():
            return 0
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        return free

    def _probe_batch(model, img_size, batch_size, num_classes, scaler):
        try:
            x = torch.randn(batch_size, 3, img_size, img_size, device=DEVICE)
            y = torch.zeros(batch_size, img_size, img_size, dtype=torch.long, device=DEVICE)
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                out  = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD(model.parameters(), lr=1e-3))
            scaler.update()
            model.zero_grad(set_to_none=True)
            del x, y, out, loss
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    # BATCH_SIZE will be determined AFTER datasets are built
    # so that DataLoader pin_memory and dataset overhead are already
    # accounted for in the free VRAM measurement.
    IMG_HEIGHT = IMG_WIDTH = IMG_SIZE

    # ── Dataset ──────────────────────────────────────────────────────
    class DentalXrayDataset(Dataset):
        def __init__(self, coco, images_dir, cat_id_to_index, img_size=(256, 256)):
            self.coco            = coco
            self.images_dir      = Path(images_dir)
            self.cat_id_to_index = cat_id_to_index
            self.img_size        = img_size
            self.image_ids       = coco.getImgIds()

        def __len__(self): return len(self.image_ids)

        def _load_mask(self, image_id, h, w):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns    = self.coco.loadAnns(ann_ids)
            mask    = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                val = self.cat_id_to_index.get(ann['category_id'], 0)
                if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        if len(seg) >= 6:
                            poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                            cv2.fillPoly(mask, [poly.astype(np.int32)], val)
            return mask

        def __getitem__(self, idx):
            image_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(image_id)[0]
            img_path = self.images_dir / img_info['file_name']
            image    = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]
            image = cv2.resize(image, self.img_size).astype(np.float32) / 255.0
            mask  = self._load_mask(image_id, orig_h, orig_w)
            mask  = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
            mask  = torch.from_numpy(np.ascontiguousarray(mask)).long()
            return image, mask

    train_dataset = DentalXrayDataset(train_coco, TRAIN_IMAGES, cat_id_to_index,
                                      img_size=(IMG_WIDTH, IMG_HEIGHT))
    val_dataset   = DentalXrayDataset(val_coco,   VAL_IMAGES,   cat_id_to_index,
                                      img_size=(IMG_WIDTH, IMG_HEIGHT))
    print(f"Train images : {len(train_dataset)}")
    print(f"Val   images : {len(val_dataset)}")

    # ── Class weights + sampler ──────────────────────────────────────
    def compute_class_weights(dataset, num_classes, num_samples=100):
        counts = np.zeros(num_classes, dtype=np.float64)
        n      = min(num_samples, len(dataset))
        for i in range(n):
            _, mask = dataset[i]
            unique, c = np.unique(mask.numpy(), return_counts=True)
            for u, cnt in zip(unique, c):
                if u < num_classes:
                    counts[u] += cnt
        weights = np.where(counts > 0, 1.0 / counts, 0.0)
        weights = weights / (weights.sum() + 1e-7) * num_classes
        weights = np.clip(weights, 0.1, 10.0)
        print(f"  Class weights: {np.round(weights, 3)}")
        return torch.FloatTensor(weights).to(DEVICE)

    def build_weighted_sampler(dataset, cat_id_to_index, coco):
        ann_by_img = {}
        for ann in coco.dataset['annotations']:
            ann_by_img.setdefault(ann['image_id'], []).append(ann['category_id'])
        all_cat_ids = [c for cats in ann_by_img.values() for c in cats]
        total = len(all_cat_ids)
        freq  = {c: all_cat_ids.count(c) / total for c in set(all_cat_ids)}
        weights = []
        for img_id in dataset.image_ids:
            cats = ann_by_img.get(img_id, [])
            weights.append(1.0 if not cats else
                           1.0 / (min(freq.get(c, 1.0) for c in cats) + 1e-9))
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights),
                                        num_samples=len(dataset), replacement=True)
        print(f"  Sampler built for {len(dataset)} images.")
        return sampler

    print("\nComputing class weights...")
    class_weights    = compute_class_weights(train_dataset, NUM_CLASSES)
    weighted_sampler = build_weighted_sampler(train_dataset, cat_id_to_index, train_coco)

    # ── Auto-tune batch size (measured HERE, after all setup overhead) ────────
    # Measuring free VRAM here is accurate because:
    #   - model weights are already on GPU
    #   - class weight tensors are on GPU
    #   - no DataLoader workers have been spawned yet
    # We use a conservative 0.70 safety factor (vs 0.85 previously) to leave
    # headroom for DataLoader pin_memory buffers and CUDA fragmentation.
    if not torch.cuda.is_available():
        BATCH_SIZE = 4
    else:
        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.synchronize()
        free_now   = _free_vram()
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"\n[AutoTune] GPU  : {torch.cuda.get_device_name(0)}")
        print(f"[AutoTune] VRAM : {_bytes_to_gb(total_vram):.2f} GB total  |  {_bytes_to_gb(free_now):.2f} GB free (after model + class weights)")

        # Warm up model BN buffers with a tiny forward pass
        _dummy = torch.randn(1, 3, 64, 64, device=DEVICE)
        with torch.no_grad():
            model(_dummy)
        del _dummy
        torch.cuda.empty_cache(); torch.cuda.synchronize()

        # Budget: 70% of free VRAM to leave room for DataLoader pin_memory
        budget = int(_free_vram() * 0.70)
        h_w    = IMG_SIZE * IMG_SIZE
        # Heuristic: bytes per image (input fp16 + activations + gradients)
        bytes_per_img = int(h_w * (3 + NUM_CLASSES + 64) * 2 * 2 * 0.6)
        hi_est = max(1, budget // bytes_per_img)

        scaler_probe = torch.amp.GradScaler('cuda', enabled=True)
        model.train()
        lo, hi, BATCH_SIZE = 1, min(hi_est, 32), 1
        print(f"[AutoTune] Budget : {_bytes_to_gb(budget):.2f} GB  (×0.70 safety factor)")
        print(f"[AutoTune] Probing batch for {IMG_SIZE}×{IMG_SIZE} (range 1..{hi})...")
        while lo <= hi:
            mid = (lo + hi) // 2
            if _probe_batch(model, IMG_SIZE, mid, NUM_CLASSES, scaler_probe):
                BATCH_SIZE = mid; lo = mid + 1
            else:
                hi = mid - 1
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        print(f"[AutoTune] ✓  batch_size={BATCH_SIZE}  (img_size={IMG_SIZE})")
    # ─────────────────────────────────────────────────────────────────────────

    _nw = 0 if platform.system() == "Windows" else min(4, max(1, os.cpu_count() // 2))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler,
                              num_workers=_nw, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=_nw, pin_memory=torch.cuda.is_available())

    # ── Loss ─────────────────────────────────────────────────────────
    class DiceLoss(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.nc = nc
        def forward(self, pred, target):
            pred = torch.softmax(pred, dim=1)
            toh  = F.one_hot(target, self.nc).permute(0, 3, 1, 2).float()
            num  = 2 * (pred * toh).sum(dim=(2, 3))
            den  = (pred + toh).sum(dim=(2, 3))
            return 1 - (num / (den + 1e-7)).mean()

    class CombinedLoss(nn.Module):
        def __init__(self, nc, cw):
            super().__init__()
            self.ce   = nn.CrossEntropyLoss(weight=cw)
            self.dice = DiceLoss(nc)
        def forward(self, pred, target):
            return self.ce(pred, target) + self.dice(pred, target)

    criterion = CombinedLoss(NUM_CLASSES, class_weights)

    # ── Optimiser + scheduler ────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    _n_train  = len(train_dataset)

    if _n_train < 1000:
        _WU  = max(3, min(10, _n_train // 50))
        _LRM = 5e-5
        def _lr_lambda(epoch):
            if epoch < _WU:
                return (epoch + 1) / _WU
            p = (epoch - _WU) / max(1, EPOCHS - _WU)
            return _LRM / LEARNING_RATE + (1 - _LRM / LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * p))
        scheduler     = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        _sched_name   = f"CosineAnnealing + warmup ({_WU} epochs)"
        _plateau_mode = False
    else:
        scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)
        _sched_name   = "ReduceLROnPlateau"
        _plateau_mode = True

    print(f"[INFO] Scheduler  : {_sched_name}  ({_n_train} train images)")

    # ── Metrics ──────────────────────────────────────────────────────
    def pixel_accuracy(pred, target):
        return (torch.argmax(pred, dim=1) == target).float().mean().item()

    def mean_iou(pred, target, nc):
        pred_cls = torch.argmax(pred, dim=1)
        ious = []
        for cls in range(nc):
            p = (pred_cls == cls); t = (target == cls)
            inter = (p & t).float().sum()
            union = (p | t).float().sum()
            if union > 0:
                ious.append((inter / union).item())
        return float(np.mean(ious)) if ious else 0.0

    # ── Training loop ────────────────────────────────────────────────
    scaler       = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    best_val_iou = 0.0
    patience_ctr = 0
    history      = {'train_loss': [], 'train_acc': [],
                    'val_loss': [],   'val_acc': [],  'val_iou': []}

    print(f"\n{'='*60}")
    print("Starting UNet Training")
    print(f"  img_size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"  batch_size : {BATCH_SIZE}")
    print(f"  epochs     : {EPOCHS}")
    print(f"  scheduler  : {_sched_name}")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        # train
        model.train()
        tl, ta = 0.0, 0.0
        for images, masks in tqdm(train_loader, desc=f"[UNet] Train {epoch+1}/{EPOCHS}", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=scaler.is_enabled()):
                out  = model(images)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            tl += loss.item(); ta += pixel_accuracy(out, masks)
        tl /= len(train_loader); ta /= len(train_loader)

        # validate
        model.eval()
        vl, va, vi = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[UNet] Val   {epoch+1}/{EPOCHS}", leave=False):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                out  = model(images)
                loss = criterion(out, masks)
                vl += loss.item(); va += pixel_accuracy(out, masks)
                vi += mean_iou(out, masks, NUM_CLASSES)
        vl /= len(val_loader); va /= len(val_loader); vi /= len(val_loader)

        if _plateau_mode: scheduler.step(vi)
        else:             scheduler.step()

        for k, v in zip(['train_loss','train_acc','val_loss','val_acc','val_iou'],
                        [tl, ta, vl, va, vi]):
            history[k].append(v)

        print(f"[UNet] Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={tl:.4f} acc={ta:.4f}  "
              f"val_loss={vl:.4f} acc={va:.4f} mIoU={vi:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if vi > best_val_iou:
            best_val_iou = vi
            patience_ctr = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_iou': vi, 'num_classes': NUM_CLASSES,
                        'cat_id_to_index': cat_id_to_index,
                        'img_size': IMG_SIZE, 'batch_size': BATCH_SIZE},
                       str(MODELS_DIR / 'best_unet.pth'))
            export_model("best_unet", model)
            print(f"  [SAVED] best_unet  mIoU={vi:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    export_model("final_unet", model)
    export_model_grads("final_unet", model.state_dict(), grads=False)
    export_model_grads("final_unet",
                       {k: v.grad for k, v in model.named_parameters() if v.grad is not None},
                       grads=True)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history['train_acc'],  label='Train'); axes[1].plot(history['val_acc'],  label='Val')
    axes[1].set_title('Pixel Accuracy'); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(history['val_iou'], color='green', label='Val mIoU')
    axes[2].set_title('Validation mIoU'); axes[2].legend(); axes[2].grid(True)
    fig.suptitle(f"UNet  |  img={IMG_SIZE}×{IMG_SIZE}  batch={BATCH_SIZE}")
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / 'unet_training_history.png'), dpi=150)
    plt.close()
    print(f"\n[UNet] Best val mIoU : {best_val_iou:.4f}")
    print(f"[UNet] Plot saved    : {MODELS_DIR / 'unet_training_history.png'}")


# ═════════════════════════════════════════════════════════════════════════════
#
#   PART 2 — ImplantClassifier  (EfficientNet + GNN)
#
# ═════════════════════════════════════════════════════════════════════════════

def train_implant_classifier():
    _section("PART 2 — ImplantClassifier Training  (EfficientNet + GNN)")

    # ── Sanity checks ────────────────────────────────────────────────
    train_csv = TRAIN_DIR / "features_augmented.csv"
    val_csv   = VAL_DIR   / "features.csv"

    for p, label in [(train_csv, "train/features_augmented.csv"),
                     (val_csv,   "val/features.csv")]:
        if not p.exists():
            print(f"[X] {label} not found at: {p}")
            if "augmented" in str(p):
                print("    Run augmentation.py first.")
            else:
                print("    Run keypoints_extraction.py first.")
            raise SystemExit(1)

    # ── Load CSVs ────────────────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # Filter val to only rows with images on disk
    val_img_dir = VAL_DIR / "cropped_images"
    existing    = set(p.stem for p in val_img_dir.glob("*.png"))
    val_df      = val_df[val_df["image_name"].isin(existing)].reset_index(drop=True)
    print(f"Train samples : {len(train_df)}")
    print(f"Val   samples : {len(val_df)}  (filtered to existing images)")

    # ── Preprocessing ────────────────────────────────────────────────
    scalar_cols = ['distance_cm', 'pixel_per_cm', 'bbox_width', 'bbox_height',
                   'implant_bbox_ratio', 'angle_left_valley', 'angle_right_valley']
    scaler = StandardScaler()
    train_df[scalar_cols] = scaler.fit_transform(train_df[scalar_cols])
    val_df[scalar_cols]   = scaler.transform(val_df[scalar_cols])

    brands      = sorted(train_df['brand'].unique())
    diameters   = sorted(train_df['implant_diameter_mm'].unique())
    brand_map   = {b: i for i, b in enumerate(brands)}
    diameter_map= {d: i for i, d in enumerate(diameters)}
    print(f"Brands    ({len(brands)})    : {brands}")
    print(f"Diameters ({len(diameters)}) : {diameters}")

    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ── Dataset ──────────────────────────────────────────────────────
    class ImplantDataset(Dataset):
        def __init__(self, df, img_dir, mask_dir, brand_map, diameter_map, transform=None):
            self.data        = df
            self.img_dir     = img_dir
            self.mask_dir    = mask_dir
            self.transform   = transform
            self.brand_map   = brand_map
            self.diameter_map= diameter_map
            self.scalar_cols = scalar_cols

        def __len__(self): return len(self.data)

        def __getitem__(self, idx):
            def _parse_node(val):
                s = str(val).strip()
                s = re.sub(r'\bnan\b', '0.0', s, flags=re.IGNORECASE)
                s = re.sub(r'\binf\b', '0.0', s, flags=re.IGNORECASE)
                try:
                    r = ast.literal_eval(s)
                    if isinstance(r, tuple):
                        return [float(r[0]), float(r[1])]
                    else:
                        # scalar — pad to (value, 0.0)
                        return [float(r), 0.0]
                except (ValueError, SyntaxError):
                    return [0.0, 0.0]

            row       = self.data.iloc[idx]
            img_path  = os.path.join(self.img_dir,  f"{row['image_name']}.png")
            mask_path = os.path.join(self.mask_dir, f"{row['image_name']}_seg.png")

            image = Image.open(img_path).convert("RGB")
            mask  = Image.open(mask_path).convert("L")

            image_np  = np.array(image)
            mask_np   = np.array(mask)
            masked    = Image.fromarray((image_np * (mask_np > 0)[..., None]).astype(np.uint8))

            if self.transform:
                image       = self.transform(masked)
                mask_tensor = transforms.ToTensor()(transforms.Resize((224, 224))(mask))

            node_keys  = ['left_top', 'right_top', 'left_bottom', 'right_bottom',
                          'interior_left', 'interior_right']
            node_feats = node_feats = torch.tensor(
                    [_parse_node(row[k]) for k in node_keys],   # shape (6, 2)
                    dtype=torch.float32
                )
            scalar_feat= torch.tensor([row[col] for col in self.scalar_cols], dtype=torch.float32)

            return (image, mask_tensor, node_feats, scalar_feat,
                    torch.tensor(self.brand_map[row['brand']]),
                    torch.tensor(self.diameter_map[row['implant_diameter_mm']]))

    train_dataset = ImplantDataset(train_df, TRAIN_DIR / "cropped_images",
                                   TRAIN_DIR / "masks", brand_map, diameter_map, transform)
    val_dataset   = ImplantDataset(val_df,   VAL_DIR   / "cropped_images",
                                   VAL_DIR   / "masks", brand_map, diameter_map, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=8)

    # ── Model + optimizer ────────────────────────────────────────────
    model     = ImplantClassifier(n_brands=len(brand_map),
                                  n_diameters=len(diameter_map)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── Edge index helper ────────────────────────────────────────────
    def _edge_index(n_nodes, batch_size):
        adj    = torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)
        ei, _  = dense_to_sparse(adj)
        return torch.cat([ei + b * n_nodes for b in range(batch_size)], dim=1).to(DEVICE)

    # ── Training loop ────────────────────────────────────────────────
    N_EPOCHS      = 10
    PATIENCE_IC   = 3
    best_val_acc  = 0.0
    patience_ctr  = 0
    best_path     = None

    print(f"\nStarting ImplantClassifier Training")
    print(f"  epochs  : {N_EPOCHS}")
    print(f"  patience: {PATIENCE_IC}")
    print("=" * 60)

    for epoch in range(N_EPOCHS):
        model.train()
        tr_losses, bp_t, bt_t, dp_t, dt_t = [], [], [], [], []

        for images, _, node_feats, scalar_feat, brand, diameter in tqdm(
                train_loader, desc=f"[IC] Train {epoch+1}/{N_EPOCHS}", leave=False):
            images, node_feats, scalar_feat, brand, diameter = (
                images.to(DEVICE), node_feats.to(DEVICE), scalar_feat.to(DEVICE),
                brand.to(DEVICE), diameter.to(DEVICE))
            B          = images.shape[0]
            batch_vec  = torch.arange(B).repeat_interleave(6).to(DEVICE)
            edge_index = _edge_index(6, B)

            optimizer.zero_grad()
            out_b, out_d = model(images, node_feats.view(-1, 2), edge_index, batch_vec, scalar_feat)
            loss = F.cross_entropy(out_b, brand) + F.cross_entropy(out_d, diameter)
            loss.backward(); optimizer.step()

            tr_losses.append(loss.item())
            bp_t += out_b.argmax(1).cpu().tolist(); bt_t += brand.cpu().tolist()
            dp_t += out_d.argmax(1).cpu().tolist(); dt_t += diameter.cpu().tolist()

        model.eval()
        val_losses, bp_v, bt_v, dp_v, dt_v = [], [], [], [], []
        with torch.no_grad():
            for images, _, node_feats, scalar_feat, brand, diameter in val_loader:
                images, node_feats, scalar_feat, brand, diameter = (
                    images.to(DEVICE), node_feats.to(DEVICE), scalar_feat.to(DEVICE),
                    brand.to(DEVICE), diameter.to(DEVICE))
                B          = images.shape[0]
                batch_vec  = torch.arange(B).repeat_interleave(6).to(DEVICE)
                edge_index = _edge_index(6, B)
                out_b, out_d = model(images, node_feats.view(-1, 2), edge_index, batch_vec, scalar_feat)
                val_losses.append((F.cross_entropy(out_b, brand) + F.cross_entropy(out_d, diameter)).item())
                bp_v += out_b.argmax(1).cpu().tolist(); bt_v += brand.cpu().tolist()
                dp_v += out_d.argmax(1).cpu().tolist(); dt_v += diameter.cpu().tolist()

        vda = accuracy_score(dt_v, dp_v)
        print(f"[IC] Epoch {epoch+1}/{N_EPOCHS} | "
              f"train_loss={np.mean(tr_losses):.4f}  val_loss={np.mean(val_losses):.4f} | "
              f"brand acc train={accuracy_score(bt_t,bp_t)*100:.1f}%  val={accuracy_score(bt_v,bp_v)*100:.1f}% | "
              f"diam  acc train={accuracy_score(dt_t,dp_t)*100:.1f}%  val={vda*100:.1f}%")

        if vda > best_val_acc:
            best_val_acc = vda
            patience_ctr = 0
            ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname     = f"EfficientNetGNN_best_valDiam_{vda:.4f}_{ts}.pt"
            best_path = MODELS_DIR / fname
            torch.save(model.state_dict(), str(best_path))
            print(f"  [SAVED] {fname}  (val_diam_acc={vda:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement. Patience: {patience_ctr}/{PATIENCE_IC}")
            if patience_ctr >= PATIENCE_IC:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\n[IC] Best val diameter acc : {best_val_acc:.4f}")
    if best_path:
        print(f"[IC] Best model saved      : {best_path}")

    export_model("final_efficientnet", model)
    export_model_grads("final_efficientnet", model.state_dict(), grads=False)
    export_model_grads("final_efficientnet",
                       {k: v.grad for k, v in model.named_parameters() if v.grad is not None},
                       grads=True)
    
    return best_path


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("█" * 70)
    print("  UNIFIED TRAINING PIPELINE")
    print(f"  Run   : {_run_name}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {MODELS_DIR}")
    print("█" * 70)

    # ── Part 1: UNet ──────────────────────────────────────────────────
    train_unet()

    # ── Free GPU memory before Part 2 ────────────────────────────────
    torch.cuda.empty_cache()
    gc.collect()

    # ── Part 2: ImplantClassifier ─────────────────────────────────────
    train_implant_classifier()

    print()
    print("█" * 70)
    print("  ALL TRAINING COMPLETE")
    print(f"  Models saved to: {MODELS_DIR}")
    print("█" * 70)