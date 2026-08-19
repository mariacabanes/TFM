"""
COCO Dataset Splitter
Dental X-ray edition.

Split : 70% train / 15% val / 15% test

The original files are NEVER modified.

Usage:
    python split_only.py

Output folder is created automatically next to this script:
    outputs/split/
        train/
            images/              <- original images
            annotations.json
        val/
            images/              <- original images
            annotations.json
        test/
            images/              <- original images
            annotations.json
"""

import json
import shutil
import numpy as np
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────
#  COLOR SYSTEM
# ─────────────────────────────────────────────

USE_COLOR = True

class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED   = "\033[91m"; GREEN  = "\033[92m"; YELLOW = "\033[93m"
    BLUE  = "\033[94m"; MAGENTA= "\033[95m"; CYAN   = "\033[96m"
    GRAY  = "\033[90m"

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

W_NAME = 28

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


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

def load_coco(annotations_path):
    print(f"\n{bold('[INFO]')} Loading: {cyan(str(annotations_path))}")
    with open(annotations_path) as f:
        raw = json.load(f)
    images_map  = {img["id"]: img for img in raw.get("images", [])}
    categories  = raw.get("categories", [])
    annotations = raw.get("annotations", [])
    print(f"       Images      : {bold(str(len(images_map)))}")
    print(f"       Categories  : {bold(str(len(categories)))}")
    print(f"       Annotations : {bold(str(len(annotations)))}")
    return raw, images_map, categories, annotations


def load_analysis(analysis_path):
    print(f"\n{bold('[INFO]')} Loading analysis results: {cyan(str(analysis_path))}")
    with open(analysis_path) as f:
        data = json.load(f)
    cleaned = data.get("cleaned_annotations", None)
    if cleaned:
        print(f"       Cleaned JSON : {cyan(cleaned)}")
    return cleaned, data


# ─────────────────────────────────────────────
#  SPLIT  70 / 15 / 15
# ─────────────────────────────────────────────

def split_ids(image_ids, seed=42):
    ids = sorted(image_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n       = len(ids)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train   = ids[:n_train]
    val     = ids[n_train : n_train + n_val]
    test    = ids[n_train + n_val :]
    return train, val, test


def build_subset(raw, image_ids, annotations):
    id_set = set(image_ids)
    return {
        "info":        raw.get("info", {}),
        "licenses":    raw.get("licenses", []),
        "categories":  raw["categories"],
        "images":      [img for img in raw["images"] if img["id"] in id_set],
        "annotations": [ann for ann in annotations  if ann["image_id"] in id_set],
    }


# ─────────────────────────────────────────────
#  COPY ORIGINAL IMAGES
# ─────────────────────────────────────────────

def copy_images(image_ids, images_map, src_dir, dst_dir):
    copied  = 0
    missing = []
    for img_id in image_ids:
        fname = images_map[img_id]["file_name"]
        src   = Path(src_dir) / Path(fname).name
        dst   = dst_dir / Path(fname).name
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(fname)
    return copied, missing


# ─────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────

def print_stats(label, coco, color_fn=green):
    ann_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    cats       = {cat["id"]: cat["name"] for cat in coco["categories"]}
    print(f"\n  {bold(label)}")
    print(f"  {gray(pad('Category', 26))}  {gray(pad('Annotations', 12, '>'))}")
    separator()
    for cat_id, name in sorted(cats.items(), key=lambda x: x[1]):
        cnt = ann_counts.get(cat_id, 0)
        print(f"  {pad(name, 26)}  {color_fn(pad(str(cnt), 12, '>'))}")
    separator()
    total = sum(ann_counts.values())
    print(f"  {bold(pad('TOTAL', 26))}  {color_fn(bold(pad(str(total), 12, '>')))}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    seed = 42
    np.random.seed(seed)

    _here         = Path(__file__).parent.resolve()
    output_dir    = _here / "outputs"
    analysis_path = output_dir / "analysis_results.json"

    if not analysis_path.exists():
        print(f"\n[X] analysis_results.json not found at: {analysis_path}")
        print("    Run dataset_analysis.py first.")
        raise SystemExit(1)

    cleaned_ann_path, analysis_data = load_analysis(str(analysis_path))

    annotations_file = cleaned_ann_path or analysis_data["source_annotations"]
    images_dir       = Path(analysis_data["source_images"]).resolve()

    print(f"       Annotations  : {cyan(str(annotations_file))}")
    print(f"       Images dir   : {cyan(str(images_dir))}")

    # ── Output folders ───────────────────────────────────────────────
    dataset_split = output_dir / "dataset_split"
    train_img = dataset_split / "train" / "images"
    val_img   = dataset_split / "val"   / "images"
    test_img  = dataset_split / "test"  / "images"
    for d in (train_img, val_img, test_img):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load COCO ────────────────────────────────────────────────────
    raw, images_map, categories, annotations = load_coco(annotations_file)

    # ── Split ───────────────────────────────────────────────────────
    section("1 . SPLIT  70 / 15 / 15")

    train_ids, val_ids, test_ids = split_ids(list(images_map.keys()), seed)
    n = len(images_map)

    print(f"\n  Total  : {bold(str(n))} images")
    print(f"  Train  : {green(bold(str(len(train_ids))))}  ({len(train_ids)/n*100:.1f}%)")
    print(f"  Val    : {cyan(bold(str(len(val_ids))))}   ({len(val_ids)/n*100:.1f}%)")
    print(f"  Test   : {yellow(bold(str(len(test_ids))))}   ({len(test_ids)/n*100:.1f}%)")

    train_coco = build_subset(raw, train_ids, annotations)
    val_coco   = build_subset(raw, val_ids,   annotations)
    test_coco  = build_subset(raw, test_ids,  annotations)

    print_stats("Train annotation distribution:", train_coco, green)
    print_stats("Val annotation distribution:",   val_coco,   cyan)
    print_stats("Test annotation distribution:",  test_coco,  yellow)

    # ── Annotations per image stats ──────────────────────────────────
    section("2 . ANNOTATIONS PER IMAGE STATISTICS  (train set)")

    train_ann_counts = Counter(ann["image_id"] for ann in train_coco["annotations"])
    all_counts = list(train_ann_counts.values())
    if all_counts:
        arr = np.array(all_counts)
        print(f"\n  Min annotations per image  : {bold(str(arr.min()))}")
        print(f"  Max annotations per image  : {bold(str(arr.max()))}")
        print(f"  Mean                       : {bold(f'{arr.mean():.2f}')}")
        print(f"  Median                     : {bold(f'{np.median(arr):.2f}')}")
        print(f"  Std                        : {bold(f'{arr.std():.2f}')}")
        buckets = [(1,1),(2,3),(4,6),(7,10),(11,20),(21,50),(51,9999)]
        print(f"\n  Distribution:")
        for lo, hi in buckets:
            cnt   = int(((arr >= lo) & (arr <= hi)).sum())
            label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 9999 else f"{lo}+")
            bar   = green("*") * min(cnt, 40)
            print(f"    {gray(pad(label, 8, '>'))} annotations/img  :  {bold(pad(str(cnt), 5, '>'))} images  {bar}")

    # ── Copy originals ───────────────────────────────────────────────
    section("3 . COPYING ORIGINAL IMAGES")

    for label, ids, dst in [
        ("Train", train_ids, train_img),
        ("Val",   val_ids,   val_img),
        ("Test",  test_ids,  test_img),
    ]:
        copied, missing = copy_images(ids, images_map, images_dir, dst)
        print(f"  {ok(f'{label}: {copied} images copied.')}")
        for m in missing:
            print(f"  {warn(f'Missing: {m}')}")

    # ── Save JSONs ───────────────────────────────────────────────────
    section("4 . SAVING ANNOTATION JSONs")

    splits = [
        ("train", train_coco, green),
        ("val",   val_coco,   cyan),
        ("test",  test_coco,  yellow),
    ]
    for split_name, coco_data, col in splits:
        split_path = dataset_split / split_name / "annotations.json"
        with open(split_path, "w") as f:
            json.dump(coco_data, f)
        print(f"\n  {ok(f'{split_name.upper()} -> {split_path}')}")
        print(f"     Images      : {col(bold(str(len(coco_data['images']))))}")
        print(f"     Annotations : {col(bold(str(len(coco_data['annotations']))))}")

    # ── Summary ──────────────────────────────────────────────────────
    section("DONE")
    separator("=")
    print(f"\n  Output folder : {cyan(str(dataset_split))}")
    print(f"""
  {bold('Folder structure created:')}

    split/
        train/  images/  ({len(train_ids)} images)  annotations.json
        val/    images/  ({len(val_ids)} images)  annotations.json
        test/   images/  ({len(test_ids)} images)  annotations.json

  {bold('Next step — augment the train set:')}

    {cyan('python augment_only.py')}

  {bold('Or train directly:')}

    {cyan('python train_local.py')}
""")
    separator("=")


if __name__ == "__main__":
    main()