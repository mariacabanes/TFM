"""
Augmentation Pipeline
=====================
Runs two augmentation stages in order:

  STAGE 1 — Crop augmentation  (geometric: flip, rotate, translate)
             Operates on cropped_images/ + masks/ + features.csv
             Transforms keypoints mathematically.
             Only applied to minor/rare diameter classes.
             Output: features_augmented.csv

  STAGE 2 — Raw image augmentation  (intensity only: brightness, gamma,
             clahe, noise, sharpness — NO flip, NO rotation to avoid
             duplicating the geometric augmentation already done in Stage 1)
             Operates on split/train/images/ and updates annotations.json.

Run AFTER:
    split_only.py
    keypoints_extraction.py

Usage:
    python augmentation.py
"""

import os
import cv2
import json
import copy
import math
import random
import numpy as np
import pandas as pd
from ast import literal_eval
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

try:
    from scipy.ndimage import map_coordinates, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────

_here         = Path(__file__).parent.resolve()
OUTPUT_DIR    = _here / "outputs"
SPLIT_DIR     = OUTPUT_DIR / "dataset_split"
TRAIN_DIR     = SPLIT_DIR / "train"

# Stage 1 paths
CSV_PATH      = TRAIN_DIR / "features.csv"
IMG_DIR       = TRAIN_DIR / "cropped_images"
MASK_DIR      = TRAIN_DIR / "masks"
OUTPUT_CSV    = TRAIN_DIR / "features_augmented.csv"

# Stage 2 paths
ANALYSIS_PATH = OUTPUT_DIR / "analysis_results.json"
TRAIN_ANN     = TRAIN_DIR / "annotations.json"
TRAIN_IMG_DIR = TRAIN_DIR / "images"

# Stage 1 settings
SEED          = 42
MINOR_CLASSES = [3.0, 3.3, 5.0, 5.5]   # diameter classes to oversample
SAVE_SUFFIXES = ["_flipH", "_flipV", "_rot90", "_rot270"]
RANDOM_ROT_SUFFIX = "_rotRand"

random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
#  COLOR SYSTEM
# ─────────────────────────────────────────────

USE_COLOR = True

class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED   = "\033[91m"; GREEN  = "\033[92m"; YELLOW = "\033[93m"
    MAGENTA= "\033[95m"; CYAN   = "\033[96m"; GRAY  = "\033[90m"

def c(code, text):
    return f"{code}{text}{C.RESET}" if USE_COLOR else text

def bold(t):    return c(C.BOLD,    t)
def red(t):     return c(C.RED,     t)
def green(t):   return c(C.GREEN,   t)
def yellow(t):  return c(C.YELLOW,  t)
def cyan(t):    return c(C.CYAN,    t)
def gray(t):    return c(C.GRAY,    t)
def magenta(t): return c(C.MAGENTA, t)

def ok(t):   return green(f"[OK]  {t}")
def info(t): return cyan(f"[i]   {t}")
def warn(t): return yellow(f"[!]   {t}")

def pad(text, width, align="<"):
    return f"{{:{align}{width}}}".format(str(text))

def section(title):
    print()
    if USE_COLOR:
        print(f"\033[44m\033[97m\033[1m  {title:<70}\033[0m")
    else:
        print("=" * 72)
        print(f"  {title}")
        print("=" * 72)

def separator(char="-", width=72):
    print(gray(char * width))


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — CROP AUGMENTATION  (geometric, keypoint-aware)
# ═════════════════════════════════════════════════════════════════════════════

def clip_coords(coords, width, height):
    return [(max(0, min(x, width - 1)), max(0, min(y, height - 1)))
            for (x, y) in coords]


def translate_image_and_coords(image, mask, coords):
    max_dx = int(image.width * 0.05)
    max_dy = int(image.height * 0.05)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    image_t = TF.affine(image, angle=0, translate=[dx, dy], scale=1.0, shear=[0, 0], fill=0)
    mask_t  = TF.affine(mask,  angle=0, translate=[dx, dy], scale=1.0, shear=[0, 0], fill=0)
    coords_t = [(x + dx, y + dy) for (x, y) in coords]
    coords_t = clip_coords(coords_t, image.width, image.height)
    return image_t, mask_t, coords_t


def augment_crop(image, mask, coords, mode):
    """Apply a geometric augmentation to a crop and transform its keypoints."""
    w, h = image.size

    if mode == "flipH":
        image  = TF.hflip(image)
        mask   = TF.hflip(mask)
        coords = [(w - x, y) for (x, y) in coords]

    elif mode == "flipV":
        image  = TF.vflip(image)
        mask   = TF.vflip(mask)
        coords = [(x, h - y) for (x, y) in coords]

    elif mode == "rot90":
        image  = image.rotate(90, expand=True)
        mask   = mask.rotate(90, expand=True)
        coords = [(y, w - x) for (x, y) in coords]

    elif mode == "rot270":
        image  = image.rotate(270, expand=True)
        mask   = mask.rotate(270, expand=True)
        coords = [(h - y, x) for (x, y) in coords]

    elif mode == "rotRand":
        angle        = random.uniform(-15, 15)
        image_r      = image.rotate(angle, resample=Image.BILINEAR, expand=True)
        mask_r       = mask.rotate(angle,  resample=Image.NEAREST,  expand=True)
        w_old, h_old = image.size
        w_new, h_new = image_r.size
        cx_old, cy_old = w_old / 2, h_old / 2
        cx_new, cy_new = w_new / 2, h_new / 2
        angle_rad = math.radians(-angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        new_coords = []
        for x, y in coords:
            xs, ys = x - cx_old, y - cy_old
            xr = xs * cos_a - ys * sin_a
            yr = xs * sin_a + ys * cos_a
            new_coords.append((xr + cx_new, yr + cy_new))
        image, mask, coords = image_r, mask_r, new_coords

    return image, mask, coords


def _safe_coord(val):
    """Parse a coordinate string that may contain nan."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.lower() in ("none", "nan", ""):
        return None
    # Replace bare nan inside tuples
    import re
    s = re.sub(r'\bnan\b', '0', s, flags=re.IGNORECASE)
    try:
        return literal_eval(s)
    except Exception:
        return None


def run_crop_augmentation():
    section("STAGE 1 — CROP AUGMENTATION  (geometric + keypoint-aware)")

    # ── Sanity checks ────────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"\n  {red('[X]')} features.csv not found at: {CSV_PATH}")
        print("       Run keypoints_extraction.py first.")
        raise SystemExit(1)
    if not IMG_DIR.exists():
        print(f"\n  {red('[X]')} cropped_images/ not found at: {IMG_DIR}")
        raise SystemExit(1)

    df = pd.read_csv(CSV_PATH)

    # Detect coordinate columns (those that look like tuples)
    keypoints_cols = [
        col for col in df.columns
        if df[col].astype(str).str.strip().str.startswith("(").any()
    ]
    print(f"\n  CSV rows        : {bold(str(len(df)))}")
    print(f"  Keypoint cols   : {cyan(str(keypoints_cols))}")
    print(f"  Minor classes   : {yellow(str(MINOR_CLASSES))}")

    # Parse coordinate columns
    for col in keypoints_cols:
        df[col] = df[col].apply(_safe_coord)

    augmented_rows = []
    skipped = 0

    print()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Augmenting crops"):

        if row["implant_diameter_mm"] not in MINOR_CLASSES:
            continue

        image_name = row["image_name"]
        img_path   = IMG_DIR  / f"{image_name}.png"
        mask_path  = MASK_DIR / f"{image_name}_seg.png"

        if not img_path.exists():
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        # Build coordinate list — None entries kept as sentinel
        original_coords = [row[col] for col in keypoints_cols]

        # Replace None with (0,0) for transform, track which were None
        safe_coords = [(0, 0) if c is None else c for c in original_coords]

        def save_aug(new_img, new_mask, new_coords, suffix):
            new_name = f"{image_name}{suffix}"
            new_img.save( IMG_DIR  / f"{new_name}.png")
            new_mask.save(MASK_DIR / f"{new_name}_seg.png")
            new_row = row.copy()
            new_row["image_name"] = new_name
            for i, col in enumerate(keypoints_cols):
                # Restore None where original was None
                new_row[col] = str(new_coords[i]) if original_coords[i] is not None else None
            augmented_rows.append(new_row)

        # 1. Fixed geometric augmentations
        for mode, suffix in zip(["flipH", "flipV", "rot90", "rot270"], SAVE_SUFFIXES):
            ni, nm, nc = augment_crop(image.copy(), mask.copy(), safe_coords.copy(), mode)
            save_aug(ni, nm, nc, suffix)

        # 2. Random rotation
        ni, nm, nc = augment_crop(image.copy(), mask.copy(), safe_coords.copy(), "rotRand")
        save_aug(ni, nm, nc, RANDOM_ROT_SUFFIX)

        # 3. Compound augmentations for specific brands
        brand_upper = str(row.get("brand", "")).upper()
        if brand_upper in ["MKIV", "SPEEDY", "PARALLEL"]:
            combos = [
                ("flipH", "rotRand",   "_flipH_rotRand"),
                ("flipV", "translate", "_flipV_translate"),
                ("flipV", "rot90",     "_flipV_rot90"),
                ("flipV", "rot270",    "_flipV_rot270"),
            ]
            for base_m, extra_m, suffix in combos:
                it, mt, ct = augment_crop(image.copy(), mask.copy(), safe_coords.copy(), base_m)
                if extra_m == "translate":
                    it, mt, ct = translate_image_and_coords(it, mt, ct)
                else:
                    it, mt, ct = augment_crop(it, mt, ct, extra_m)
                save_aug(it, mt, ct, suffix)

        # 4. Random translation
        ni, nm, nc = translate_image_and_coords(image.copy(), mask.copy(), safe_coords.copy())
        save_aug(ni, nm, nc, "_translate")

    # Save combined CSV
    augmented_df = pd.DataFrame(augmented_rows)
    df_final     = pd.concat([df, augmented_df], ignore_index=True)
    df_final.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  {ok(f'Original samples  : {len(df)}')}")
    print(f"  {ok(f'New samples added : {len(augmented_rows)}')}")
    print(f"  {ok(f'Total samples     : {len(df_final)}')}")
    if skipped:
        print(f"  {warn(f'Skipped (missing) : {skipped}')}")
    print(f"  {ok(f'CSV saved -> {OUTPUT_CSV}')}")


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — RAW IMAGE AUGMENTATION  (intensity only, no geometry)
# ═════════════════════════════════════════════════════════════════════════════

# ── Intensity-only pipeline (NO flip, NO rotation) ───────────────────────────

def aug_brightness(image, masks, low=0.75, high=1.25):
    factor = np.random.uniform(low, high)
    img    = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img, masks

def aug_gamma(image, masks, low=0.75, high=1.25):
    gamma = np.random.uniform(low, high)
    lut   = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut), masks

def aug_gaussian_noise(image, masks, sigma_range=(3, 8)):
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    img   = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, masks

def aug_clahe(image, masks, clip_range=(1.5, 3.0)):
    clip = np.random.uniform(*clip_range)
    lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cl   = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), masks

def aug_sharpness(image, masks):
    sigma   = np.random.uniform(0.5, 1.5)
    amount  = np.random.uniform(0.3, 0.8)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    img     = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return img, masks

def aug_elastic(image, masks, alpha_range=(15, 30), sigma=5):
    if not HAS_SCIPY:
        return image, masks
    h, w   = image.shape[:2]
    alpha  = np.random.uniform(*alpha_range)
    dx     = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    dy     = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
    x, y   = np.meshgrid(np.arange(w), np.arange(h))
    coords = [np.clip(y + dy, 0, h - 1).ravel(),
              np.clip(x + dx, 0, w - 1).ravel()]
    def distort(arr, order):
        if arr.ndim == 3:
            return np.stack([
                map_coordinates(arr[:, :, ch], coords, order=order,
                                mode="reflect").reshape(h, w)
                for ch in range(arr.shape[2])
            ], axis=2).astype(arr.dtype)
        return map_coordinates(arr, coords, order=order,
                               mode="reflect").reshape(h, w).astype(arr.dtype)
    return distort(image, 1), [distort(m, 0) for m in masks]

# Intensity-only pipeline — geometric transforms deliberately excluded
INTENSITY_PIPELINE = [
    ("brightness", aug_brightness,    0.6),
    ("gamma",      aug_gamma,         0.5),
    ("clahe",      aug_clahe,         0.5),
    ("noise",      aug_gaussian_noise,0.4),
    ("sharpness",  aug_sharpness,     0.3),
    ("elastic",    aug_elastic,       0.35),
]

def apply_intensity_pipeline(image, masks):
    for name, fn, prob in INTENSITY_PIPELINE:
        if np.random.random() < prob:
            image, masks = fn(image, masks)
    return image, masks


def anns_to_masks(anns, h, w):
    masks = []
    for ann in anns:
        mask = np.zeros((h, w), dtype=np.uint8)
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            for seg in ann["segmentation"]:
                if len(seg) >= 6:
                    poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        masks.append(mask)
    return masks

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt.flatten().tolist() for cnt in contours if cnt.size >= 6]

def polygon_area(poly):
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

def mask_bbox(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return [0, 0, 0, 0]
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]

def image_aug_factor(img_id, ann_by_img, cat_factors, default_factor=4):
    anns    = ann_by_img.get(img_id, [])
    factors = [cat_factors.get(ann["category_id"], default_factor)
               for ann in anns
               if cat_factors.get(ann["category_id"], 1) > 0]
    return max(factors) if factors else default_factor


def run_raw_augmentation():
    section("STAGE 2 — RAW IMAGE AUGMENTATION  (intensity only, no geometry)")

    # ── Sanity checks ────────────────────────────────────────────────
    for path, label in [(ANALYSIS_PATH, "analysis_results.json"),
                        (TRAIN_ANN,     "train/annotations.json"),
                        (TRAIN_IMG_DIR, "train/images/")]:
        if not path.exists():
            print(f"\n  {red('[X]')} {label} not found at: {path}")
            raise SystemExit(1)

    # ── Load analysis ────────────────────────────────────────────────
    with open(ANALYSIS_PATH) as f:
        analysis_data = json.load(f)

    cat_factors = {
        int(cid): info["aug_factor"]
        for cid, info in analysis_data["categories"].items()
    }
    images_dir = Path(analysis_data["source_images"]).resolve()

    # ── Load train annotations ───────────────────────────────────────
    with open(TRAIN_ANN) as f:
        train_coco = json.load(f)

    images_map  = {img["id"]: img for img in train_coco["images"]}
    annotations = train_coco["annotations"]
    train_ids   = list(images_map.keys())

    print(f"\n  Train images      : {bold(str(len(train_ids)))}")
    print(f"  Train annotations : {bold(str(len(annotations)))}")
    print(f"\n  {info('Intensity augmentations active:')}")
    for name, fn, prob in INTENSITY_PIPELINE:
        skip = name == "elastic" and not HAS_SCIPY
        status = yellow("[SKIP]") if skip else green("[ON]  ")
        print(f"    {status} {pad(name, 12)}  p={prob}")
    if not HAS_SCIPY:
        print(f"\n  {warn('scipy not installed — elastic distortion disabled.')}")

    # ── Build aug factor map ─────────────────────────────────────────
    ann_by_img = {}
    for ann in annotations:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    aug_factor_map = {
        img_id: image_aug_factor(img_id, ann_by_img, cat_factors)
        for img_id in train_ids
    }

    next_img_id = max(images_map.keys(), default=0) + 1
    next_ann_id = max((a["id"] for a in annotations), default=0) + 1

    new_images = []
    new_anns   = []
    skipped    = 0
    total      = len(train_ids)

    print()
    for i, img_id in enumerate(train_ids):
        img_info   = images_map[img_id]
        fname      = img_info["file_name"]
        src        = TRAIN_IMG_DIR / Path(fname).name

        # Fall back to original source if not yet copied to split dir
        if not src.exists():
            src = images_dir / Path(fname).name

        image = cv2.imread(str(src))
        if image is None:
            skipped += 1
            continue

        aug_factor = aug_factor_map[img_id]
        h, w       = image.shape[:2]
        anns       = ann_by_img.get(img_id, [])
        masks      = anns_to_masks(anns, h, w)

        pct     = (i + 1) / total
        bar_len = int(pct * 40)
        bar     = green("█" * bar_len) + gray("░" * (40 - bar_len))
        print(f"\r  [{bar}] {i+1:>3}/{total}  x{aug_factor}  {gray(Path(fname).name[:30])}",
              end="", flush=True)

        for k in range(aug_factor):
            aug_img, aug_masks = apply_intensity_pipeline(
                image.copy(), [m.copy() for m in masks]
            )

            stem     = Path(fname).stem
            aug_name = f"{stem}_intaug{k:02d}.jpg"
            cv2.imwrite(str(TRAIN_IMG_DIR / aug_name), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 93])

            new_images.append({
                "id":        next_img_id,
                "file_name": aug_name,
                "width":     w,
                "height":    h,
            })

            for ann, mask in zip(anns, aug_masks):
                bin_mask = (mask > 0).astype(np.uint8)
                polys    = mask_to_polygons(bin_mask)
                if not polys:
                    continue
                area = sum(polygon_area(p) for p in polys)
                if area < 4:
                    continue
                new_anns.append({
                    "id":           next_ann_id,
                    "image_id":     next_img_id,
                    "category_id":  ann["category_id"],
                    "segmentation": polys,
                    "area":         float(area),
                    "bbox":         mask_bbox(bin_mask),
                    "iscrowd":      0,
                })
                next_ann_id += 1

            next_img_id += 1

    print()  # newline after progress bar

    # ── Save updated annotations.json ───────────────────────────────
    final_train = copy.deepcopy(train_coco)
    final_train["images"]      += new_images
    final_train["annotations"] += new_anns

    with open(TRAIN_ANN, "w") as f:
        json.dump(final_train, f)

    orig  = len(train_ids)
    added = len(new_images)
    print(f"\n  {ok(f'Original train images   : {orig}')}")
    print(f"  {ok(f'Augmented images added  : {added}')}")
    print(f"  {ok(f'Total train images      : {orig + added}')}")
    print(f"  {ok(f'New annotations created : {len(new_anns)}')}")
    if skipped:
        print(f"  {warn(f'Images skipped          : {skipped}')}")
    print(f"  {ok(f'annotations.json updated -> {TRAIN_ANN}')}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print(bold("=" * 72))
    print(bold("  AUGMENTATION PIPELINE"))
    print(bold("  Stage 1: Crop augmentation  (geometric + keypoints)"))
    print(bold("  Stage 2: Raw image augmentation  (intensity only)"))
    print(bold("=" * 72))

    # ── Stage 1 ──────────────────────────────────────────────────────
    run_crop_augmentation()

    # ── Stage 2 ──────────────────────────────────────────────────────
    run_raw_augmentation()

    # ── Final summary ────────────────────────────────────────────────
    section("DONE")
    separator("=")
    print(f"""
  Stage 1 output : {cyan(str(OUTPUT_CSV))}
  Stage 2 output : {cyan(str(TRAIN_IMG_DIR))}
               -> {cyan(str(TRAIN_ANN))}

  {bold('Next steps:')}

    EfficientNet  :  {cyan('python efficientNet_b2_train_test.py')}
                     uses {OUTPUT_CSV.name}

    UNet / FL     :  {cyan('python train_local.py')}
                     uses the augmented images/
""")
    separator("=")


if __name__ == "__main__":
    main()