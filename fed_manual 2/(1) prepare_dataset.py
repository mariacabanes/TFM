"""
CVAT Dataset Preparator
Extracts and merges multiple CVAT COCO 1.0 zip files into a single
unified dataset folder ready for dataset_analysis.py.

Each zip is expected to follow the CVAT COCO 1.0 export structure:
    <name>.zip
        images/
            default/
                *.jpg / *.png ...
        annotations/
            instances_default.json

All zips must be placed in the same input folder before running.

Usage:
    python prepare_dataset.py --input /path/to/folder/with/zips/
                              --output /path/to/output/dataset/

Output structure:
    output/
        images/              <- all images from all zips (flat)
        annotations.json     <- merged COCO JSON

Options:
    --input   PATH   Folder containing the .zip files  (required)
    --output  PATH   Output folder  (default: <input>_merged)
    --no-color       Disable colored output
"""

import os
import json
import copy
import shutil
import zipfile
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

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
def err(t):  return red(f"[X]   {t}")

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


# ─────────────────────────────────────────────
#  ZIP DISCOVERY
# ─────────────────────────────────────────────

def find_zips(input_dir):
    zips = sorted(Path(input_dir).glob("*.zip"))
    if not zips:
        print(red(f"\n[X]  No .zip files found in: {input_dir}"))
        raise SystemExit(1)
    print(f"\n{bold('[INFO]')} Found {bold(str(len(zips)))} zip file(s) in {cyan(str(input_dir))}:")
    for z in zips:
        size_kb = z.stat().st_size // 1024
        print(f"       {green('+')} {pad(z.name, 40)}  {gray(f'{size_kb:,} KB')}")
    return zips


# ─────────────────────────────────────────────
#  ZIP INSPECTION
# ─────────────────────────────────────────────

def inspect_zip(zip_path):
    """
    Detect the internal structure of a CVAT COCO zip.
    Returns (images_prefix, annotations_prefix) inside the zip.

    CVAT COCO 1.0 exports typically look like:
        images/default/*.jpg
        annotations/instances_default.json

    But some versions use:
        images/*.jpg
        annotations/instances_default.json
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    json_files = [n for n in names if n.endswith(".json")]
    img_files  = [n for n in names
                  if Path(n).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tiff",".webp"}]

    if not json_files:
        raise ValueError(f"No JSON file found inside {zip_path.name}")
    if not img_files:
        raise ValueError(f"No image files found inside {zip_path.name}")

    json_path = json_files[0]   # e.g.  annotations/instances_default.json
    # Find the common image prefix — could be images/ or images/default/
    img_prefix = str(Path(img_files[0]).parent)

    return json_path, img_prefix, img_files


# ─────────────────────────────────────────────
#  EXTRACT ONE ZIP
# ─────────────────────────────────────────────

def extract_zip(zip_path, tmp_dir):
    """Extract zip to a temp subfolder, return that folder path."""
    dest = Path(tmp_dir) / zip_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


# ─────────────────────────────────────────────
#  LOAD ONE COCO JSON
# ─────────────────────────────────────────────

def load_json(json_abs_path):
    with open(json_abs_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  MERGE COCO DATASETS
# ─────────────────────────────────────────────

def merge_datasets(entries):
    """
    Merge a list of (coco_dict, source_name, extracted_dir, img_prefix) tuples
    into a single unified COCO dict.

    ID remapping strategy:
      - category IDs: unified across all zips (same name = same ID)
      - image IDs: remapped to avoid collisions
      - annotation IDs: remapped to avoid collisions

    Image filenames: prefixed with <source_name>__ to avoid conflicts
    between zips that may have identical filenames.
    """
    section("3 . MERGING COCO DATASETS")

    # ── Build unified category list ──────────────────────────────────
    # Collect all unique category names across all zips
    all_cat_names = {}   # name -> first-seen supercategory
    for coco, src, _, _ in entries:
        for cat in coco.get("categories", []):
            name = cat["name"]
            if name not in all_cat_names:
                all_cat_names[name] = cat.get("supercategory", "")

    unified_cats = [
        {"id": idx + 1, "name": name, "supercategory": sup}
        for idx, (name, sup) in enumerate(sorted(all_cat_names.items()))
    ]
    cat_name_to_id = {cat["name"]: cat["id"] for cat in unified_cats}

    print(f"\n  {bold('Unified categories:')}  {bold(str(len(unified_cats)))}")
    for cat in unified_cats:
        print(f"    {gray(pad(str(cat['id']), 3, '>'))}.  {cat['name']}")

    # ── Merge images and annotations ─────────────────────────────────
    merged_images      = []
    merged_annotations = []
    next_img_id  = 1
    next_ann_id  = 1

    stats = []

    for coco, src_name, extracted_dir, img_prefix in entries:
        old_cat_id_map = {
            cat["id"]: cat_name_to_id[cat["name"]]
            for cat in coco.get("categories", [])
            if cat["name"] in cat_name_to_id
        }
        old_img_id_to_new = {}
        img_count  = 0
        ann_count  = 0
        skip_imgs  = 0

        for img in coco.get("images", []):
            old_fname = img["file_name"]
            # CVAT sometimes stores as "default/img.jpg" — strip subfolder
            bare_name = Path(old_fname).name
            # Prefix with source name to avoid filename collisions
            new_fname = f"{src_name}__{bare_name}"

            old_img_id_to_new[img["id"]] = next_img_id
            merged_images.append({
                "id":        next_img_id,
                "file_name": new_fname,
                "width":     img.get("width",  0),
                "height":    img.get("height", 0),
            })
            next_img_id += 1
            img_count   += 1

        for ann in coco.get("annotations", []):
            new_img_id = old_img_id_to_new.get(ann["image_id"])
            new_cat_id = old_cat_id_map.get(ann["category_id"])
            if new_img_id is None or new_cat_id is None:
                continue
            new_ann = copy.deepcopy(ann)
            new_ann["id"]          = next_ann_id
            new_ann["image_id"]    = new_img_id
            new_ann["category_id"] = new_cat_id
            merged_annotations.append(new_ann)
            next_ann_id += 1
            ann_count   += 1

        stats.append((src_name, img_count, ann_count))

    # ── Print merge summary ──────────────────────────────────────────
    print(f"\n  {gray(pad('Source', 36))}  {gray(pad('Images', 7, '>'))  }  {gray(pad('Annotations', 11, '>'))}")
    separator()
    for src_name, img_c, ann_c in stats:
        print(f"  {pad(src_name, 36)}  {green(pad(str(img_c), 7, '>'))  }  {green(pad(str(ann_c), 11, '>'))}")
    separator()
    total_imgs = sum(s[1] for s in stats)
    total_anns = sum(s[2] for s in stats)
    print(f"  {bold(pad('TOTAL', 36))}  {bold(pad(str(total_imgs), 7, '>'))  }  {bold(pad(str(total_anns), 11, '>'))}")

    merged = {
        "info":        {"description": "Merged CVAT COCO dataset"},
        "licenses":    [],
        "categories":  unified_cats,
        "images":      merged_images,
        "annotations": merged_annotations,
    }
    return merged


# ─────────────────────────────────────────────
#  COPY IMAGES TO OUTPUT
# ─────────────────────────────────────────────

def copy_images_to_output(entries, merged_coco, output_images_dir):
    """
    For each image in the merged COCO, find the source file and copy it
    to output_images_dir with the new prefixed filename.
    """
    section("4 . COPYING IMAGES TO OUTPUT")

    # Build lookup: new_fname -> (extracted_dir, img_prefix)
    # We need to reverse-map new_fname -> source entry
    # new_fname = "<src_name>__<bare_name>"
    src_map = {src_name: (extracted_dir, img_prefix)
               for _, src_name, extracted_dir, img_prefix in entries}

    copied  = 0
    missing = []

    total = len(merged_coco["images"])
    for i, img in enumerate(merged_coco["images"]):
        new_fname  = img["file_name"]           # e.g. zimvie_tsx_4.7__img001.jpg
        # split on first __ to get source name and bare name
        parts      = new_fname.split("__", 1)
        if len(parts) != 2:
            missing.append(new_fname)
            continue
        src_name, bare_name = parts
        extracted_dir, img_prefix = src_map.get(src_name, (None, None))
        if extracted_dir is None:
            missing.append(new_fname)
            continue

        # Try images/default/<bare_name> then images/<bare_name>
        candidates = [
            extracted_dir / img_prefix / bare_name,
            extracted_dir / "images" / "default" / bare_name,
            extracted_dir / "images" / bare_name,
        ]
        src_file = next((p for p in candidates if p.exists()), None)

        if src_file is None:
            missing.append(new_fname)
            continue

        dst = output_images_dir / new_fname
        shutil.copy2(src_file, dst)
        copied += 1

        # Progress
        pct     = (i + 1) / total
        bar_len = int(pct * 40)
        bar     = green("*" * bar_len) + gray("." * (40 - bar_len))
        print(f"\r  [{bar}] {i+1:>4}/{total}", end="", flush=True)

    print()  # newline after progress
    print(f"\n  {ok(f'Copied  : {copied} images')}")
    if missing:
        print(f"  {warn(f'Missing : {len(missing)} images')}")
        for m in missing[:10]:
            print(f"    {red('[X]')} {m}")
        if len(missing) > 10:
            print(gray(f"    ... and {len(missing) - 10} more"))

    return copied, missing


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="CVAT COCO Dataset Preparator — extracts and merges CVAT COCO zip files."
    )
    parser.add_argument("input",      nargs="?", default=None,
                        help="Folder containing the .zip files "
                             "(default: same folder as this script)")
    args = parser.parse_args()

    here       = Path(__file__).parent.resolve()
    output_dir = here / "outputs"
    input_dir  = Path(args.input) if args.input else here / "input_zips"
    output_datset = output_dir / "dataset"
    output_images_dir = output_datset / "images"
    tmp_dir           = output_dir / "_tmp_extract"

    output_images_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Find zips ─────────────────────────────────────────────────
    zips = find_zips(input_dir)

    # ── 2. Extract all zips ──────────────────────────────────────────
    section("2 . EXTRACTING ZIP FILES")

    entries = []   # list of (coco_dict, src_name, extracted_dir, img_prefix)
    for zip_path in zips:
        src_name     = zip_path.stem   # e.g. "zimvie_tsx_4.7"
        print(f"\n  {info(f'Extracting {zip_path.name} ...')}")

        try:
            extracted_dir           = extract_zip(zip_path, tmp_dir)
            json_rel, img_prefix, img_files = inspect_zip(zip_path)
            json_abs                = extracted_dir / json_rel
            coco                    = load_json(json_abs)

            n_imgs = len(coco.get("images", []))
            n_anns = len(coco.get("annotations", []))
            n_cats = len(coco.get("categories", []))

            print(f"     {ok(f'Images: {n_imgs}  |  Annotations: {n_anns}  |  Categories: {n_cats}')}")
            print(f"     {gray(f'Images prefix inside zip: {img_prefix}')}")

            entries.append((coco, src_name, extracted_dir, img_prefix))

        except Exception as e:
            print(f"     {err(f'Failed: {e}')}")
            continue

    if not entries:
        print(red("\n[X]  No zips could be extracted. Aborting."))
        raise SystemExit(1)

    # ── 3. Merge ─────────────────────────────────────────────────────
    merged_coco = merge_datasets(entries)

    # ── 4. Copy images ───────────────────────────────────────────────
    copy_images_to_output(entries, merged_coco, output_images_dir)

    # ── 5. Save merged JSON ──────────────────────────────────────────
    section("5 . SAVING MERGED ANNOTATIONS JSON")

    annotations_path = output_datset / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(merged_coco, f)

    print(f"\n  {ok(f'annotations.json -> {annotations_path}')}")
    print(f"     Images      : {bold(str(len(merged_coco['images'])))}")
    print(f"     Categories  : {bold(str(len(merged_coco['categories'])))}")
    print(f"     Annotations : {bold(str(len(merged_coco['annotations'])))}")

    # ── 6. Cleanup tmp ───────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── 7. Save dataset_info.json ────────────────────────────────────
    section("6 . SAVING DATASET INFO")

    dataset_info = {
        "annotations":      str(annotations_path),
        "images_dir":       str(output_images_dir),
        "output_dir":       str(output_dir),
        "num_images":       len(merged_coco["images"]),
        "num_categories":   len(merged_coco["categories"]),
        "num_annotations":  len(merged_coco["annotations"]),
        "categories": [
            {"id": cat["id"], "name": cat["name"]}
            for cat in merged_coco["categories"]
        ],
    }
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n  {ok(f'dataset_info.json -> {info_path}')}")
    print(f"  {gray('This file is the handoff to dataset_analysis.py — no extra paths needed.')}")

    # ── Summary ──────────────────────────────────────────────────────
    section("DONE")
    separator("=")
    print(f"\n  Output folder : {cyan(str(output_dir))}")
    print(f"""
  {bold('Folder structure:')}

    {output_dir}/
        dataset/
            images/              <- all images from all {len(zips)} zip(s), prefixed by source
            annotations.json     <- unified COCO JSON
        dataset_info.json    <- handoff file  (AUTO-READ by next steps)

  {bold('Next step:')}

    {cyan('python dataset_analysis.py')}
""")
    separator("=")


if __name__ == "__main__":
    main()