"""
COCO Dataset Analyzer + Interactive Cleanup
Analyzes class distribution, imbalance, and image/annotation mismatches.
After analysis, offers to fix issues interactively and export a clean JSON.

Usage:
    python analyze_coco.py --annotations path/to/annotations.json --images path/to/images/
    python analyze_coco.py --annotations path/to/annotations.json --images path/to/images/ --chart
    python analyze_coco.py --annotations path/to/annotations.json --images path/to/images/ --no-cleanup
    python analyze_coco.py --annotations path/to/annotations.json --images path/to/images/ --no-color
"""

import os
import json
import copy
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ─────────────────────────────────────────────
#  COLOR SYSTEM
# ─────────────────────────────────────────────

USE_COLOR = True

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_BLUE = "\033[44m"

def c(code, text):
    if not USE_COLOR:
        return text
    return f"{code}{text}{C.RESET}"

def bold(text):    return c(C.BOLD,    text)
def red(text):     return c(C.RED,     text)
def green(text):   return c(C.GREEN,   text)
def yellow(text):  return c(C.YELLOW,  text)
def blue(text):    return c(C.BLUE,    text)
def magenta(text): return c(C.MAGENTA, text)
def cyan(text):    return c(C.CYAN,    text)
def gray(text):    return c(C.GRAY,    text)
def white(text):   return c(C.WHITE,   text)

# Status labels — no emojis, plain ASCII tags
def ok(text):    return green(f"[OK]  {text}")
def warn(text):  return yellow(f"[!]   {text}")
def err(text):   return red(f"[X]   {text}")
def info(text):  return cyan(f"[i]   {text}")

def pad(text, width, align="<"):
    """
    Pad a PLAIN string to a fixed width, then return the padded plain string.
    Always call this BEFORE wrapping in a color function so ANSI codes
    do not inflate the measured width and break column alignment.
    """
    return f"{{:{align}{width}}}".format(str(text))


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
W_NAME  = 28   # category name column width
W_ID    = 4    # id column width
W_COUNT = 7    # count column width
W_PCT   = 7    # percent column width
W_DIFF  = 9    # diff-vs-mean column width

def separator(char="-", width=72):
    print(gray(char * width))

def section(title):
    print()
    if USE_COLOR:
        print(f"\033[44m\033[97m\033[1m  {title:<70}\033[0m")
    else:
        print("=" * 72)
        print(f"  {title}")
        print("=" * 72)

def ask(prompt, options):
    print(f"\n  {bold(prompt)}")
    for key, desc in options:
        print(f"    {cyan(bold(f'[{key}]'))} {desc}")
    valid = {k.lower() for k, _ in options}
    while True:
        if USE_COLOR:
            choice = input(f"  \033[96m>\033[0m ").strip().lower()
        else:
            choice = input("  > ").strip().lower()
        if choice in valid:
            return choice
        print(red(f"  Invalid choice. Enter one of: {', '.join(sorted(valid))}"))

def ask_yes_no(prompt):
    return ask(prompt, [("y", "Yes"), ("n", "No")]) == "y"


# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────

def load_coco(annotations_path):
    print(f"\n{bold('[INFO]')} Loading annotations: {cyan(str(annotations_path))}")
    with open(annotations_path, "r") as f:
        raw = json.load(f)

    images      = {img["id"]: img for img in raw.get("images", [])}
    categories  = {cat["id"]: cat for cat in raw.get("categories", [])}
    annotations = raw.get("annotations", [])

    print(f"       Images in JSON   : {bold(str(len(images)))}")
    print(f"       Categories       : {bold(str(len(categories)))}")
    print(f"       Annotations      : {bold(str(len(annotations)))}")
    return images, categories, annotations, raw


def scan_image_folder(images_dir):
    folder = Path(images_dir)
    files  = {
        f.name: f
        for f in folder.rglob("*")
        if f.suffix.lower() in VALID_EXTENSIONS
    }
    print(f"\n{bold('[INFO]')} Image folder     : {cyan(str(images_dir))}")
    print(f"       Files found      : {bold(str(len(files)))}")
    return files


# ─────────────────────────────────────────────
#  ANALYSIS 1 — CLASS DISTRIBUTION
# ─────────────────────────────────────────────

def analyze_classes(annotations, categories):
    section("1 . CLASS DISTRIBUTION")

    cat_counts = Counter(ann["category_id"] for ann in annotations)
    rows = []
    for cat_id, cat_info in sorted(categories.items(), key=lambda x: x[1]["name"]):
        count = cat_counts.get(cat_id, 0)
        rows.append((cat_id, cat_info["name"], count))

    total     = sum(r[2] for r in rows)
    max_count = max(r[2] for r in rows) if rows else 1

    # Header — pad plain strings, then colorize
    h_id    = gray(pad("ID",       W_ID,    ">"))
    h_name  = gray(pad("Category", W_NAME,  "<"))
    h_count = gray(pad("Count",    W_COUNT, ">"))
    h_pct   = gray(pad("%",        W_PCT,   ">"))
    h_bar   = gray("Bar")
    print(f"\n  {h_id}  {h_name}  {h_count}  {h_pct}  {h_bar}")
    separator()

    for cat_id, name, count in sorted(rows, key=lambda x: -x[2]):
        pct     = (count / total * 100) if total > 0 else 0
        bar_len = int(count / max_count * 30)

        # Pad first, colorize after — this is the key fix for alignment
        c_id    = pad(str(cat_id),       W_ID,    ">")
        c_name  = pad(name,              W_NAME,  "<")
        c_count = pad(str(count),        W_COUNT, ">")
        c_pct   = pad(f"{pct:.1f}%",     W_PCT,   ">")

        if count == 0:
            bar = "XXXXX EMPTY"
            print(f"  {red(c_id)}  {red(c_name)}  {red(c_count)}  {red(c_pct)}  {red(bar)}")
        elif count / max_count > 0.5:
            bar = magenta("*" * bar_len)
            print(f"  {gray(c_id)}  {magenta(bold(c_name))}  {magenta(c_count)}  {magenta(c_pct)}  {bar}")
        elif count / max_count > 0.1:
            bar = blue("*" * bar_len)
            print(f"  {gray(c_id)}  {blue(c_name)}  {blue(c_count)}  {blue(c_pct)}  {bar}")
        else:
            bar = yellow("*" * max(bar_len, 1))
            print(f"  {gray(c_id)}  {yellow(c_name)}  {yellow(c_count)}  {yellow(c_pct)}  {bar}")

    separator()
    print(f"  {pad('', W_ID, '>')}  {bold(pad('TOTAL', W_NAME, '<'))}  {bold(pad(str(total), W_COUNT, '>'))}")

    return rows, cat_counts


# ─────────────────────────────────────────────
#  ANALYSIS 2 — CLASS IMBALANCE
# ─────────────────────────────────────────────

def analyze_imbalance(rows):
    section("2 . CLASS IMBALANCE REPORT")

    counts = np.array([r[2] for r in rows], dtype=float)
    names  = [r[1] for r in rows]

    if len(counts) == 0:
        print(warn("No categories found."))
        return None, [], []

    mean   = counts.mean()
    std    = counts.std()
    median = np.median(counts)

    nonzero      = counts[counts > 0]
    zero_classes = [(rows[i][0], names[i]) for i, c in enumerate(counts) if c == 0]
    ratio        = (nonzero.max() / nonzero.min()) if len(nonzero) >= 2 else 1.0

    print(f"\n  Mean annotations / class  : {bold(f'{mean:.1f}')}")
    print(f"  Median                    : {bold(f'{median:.1f}')}")
    print(f"  Std deviation             : {bold(f'{std:.1f}')}")
    print(f"  Max / Min ratio (non-zero): {bold(f'{ratio:.1f}x')}")

    if zero_classes:
        print(f"\n  {red(bold(f'[X] Classes with ZERO annotations ({len(zero_classes)}):'))}")
        for cat_id, name in zero_classes:
            print(f"      {red('*')} {red(f'[{cat_id}]')} {red(name)}")

    SEVERE_RATIO   = 10
    MODERATE_RATIO = 3

    if ratio >= SEVERE_RATIO:
        verdict_str = red(bold("[!] SEVERELY IMBALANCED"))
    elif ratio >= MODERATE_RATIO:
        verdict_str = yellow(bold("[!] MODERATELY IMBALANCED"))
    else:
        verdict_str = green(bold("[OK] BALANCED"))

    print(f"\n  Verdict: {verdict_str}")

    # Table header
    h_name   = gray(pad("Category", W_NAME,  "<"))
    h_count  = gray(pad("Count",    W_COUNT, ">"))
    h_diff   = gray(pad("vs Mean",  W_DIFF,  ">"))
    h_status = gray("Status")
    print(f"\n  {h_name}  {h_count}  {h_diff}  {h_status}")
    separator()

    low_classes = []
    for (cat_id, name, count_int), count in zip(rows, counts):
        diff_pct = ((count - mean) / mean * 100) if mean > 0 else 0
        sign     = "+" if diff_pct >= 0 else ""

        # Pad all plain values first
        c_name  = pad(name,                         W_NAME,  "<")
        c_count = pad(str(int(count)),               W_COUNT, ">")
        c_diff  = pad(f"{sign}{diff_pct:.1f}%",      W_DIFF,  ">")

        if count == 0:
            status = red("[X]  EMPTY")
            low_classes.append((cat_id, name, int(count)))
            print(f"  {red(c_name)}  {red(c_count)}  {red(c_diff)}  {status}")
        elif count < mean * 0.3:
            status = yellow("[!]  UNDER-REPRESENTED")
            low_classes.append((cat_id, name, int(count)))
            print(f"  {yellow(c_name)}  {yellow(c_count)}  {yellow(c_diff)}  {status}")
        elif count > mean * 3:
            status = magenta("[^]  OVER-REPRESENTED")
            print(f"  {magenta(bold(c_name))}  {magenta(c_count)}  {magenta(c_diff)}  {status}")
        else:
            status = green("[OK] OK")
            print(f"  {c_name}  {c_count}  {green(c_diff)}  {status}")

    return ratio, zero_classes, low_classes


# ─────────────────────────────────────────────
#  ANALYSIS 3 — IMAGES WITHOUT ANNOTATIONS
# ─────────────────────────────────────────────

def analyze_unannotated_images(images, annotations, disk_files):
    section("3 . IMAGES WITHOUT ANNOTATIONS")

    annotated_ids   = set(ann["image_id"] for ann in annotations)
    all_json_ids    = set(images.keys())
    unannotated_ids = all_json_ids - annotated_ids

    print(f"\n  Images in JSON             : {bold(str(len(all_json_ids)))}")
    print(f"  Images with annotations    : {green(bold(str(len(annotated_ids))))}")

    count_col = red(bold(str(len(unannotated_ids)))) if unannotated_ids else green(bold("0"))
    print(f"  Images WITHOUT annotations : {count_col}")

    if unannotated_ids:
        print(f"\n  {yellow('List of unannotated images:')}")
        separator()
        for img_id in sorted(unannotated_ids):
            img_info = images[img_id]
            fname    = img_info.get("file_name", "unknown")
            on_disk  = green("[on disk]") if Path(fname).name in disk_files else red("[NOT on disk]")
            print(f"    {gray(pad(str(img_id), 6, '>'))}  {cyan(fname)}  {on_disk}")
    else:
        print(f"\n  {ok('All images in the JSON have at least one annotation.')}")

    return unannotated_ids


# ─────────────────────────────────────────────
#  ANALYSIS 4 — ORPHAN ANNOTATIONS
# ─────────────────────────────────────────────

def analyze_orphan_annotations(images, annotations):
    section("4 . ANNOTATIONS POINTING TO MISSING IMAGE IDs")

    all_json_ids = set(images.keys())
    orphan_anns  = [ann for ann in annotations if ann["image_id"] not in all_json_ids]

    label = red(bold(str(len(orphan_anns)))) if orphan_anns else green(bold("0"))
    print(f"\n  Orphan annotations (no matching image ID): {label}")

    if orphan_anns:
        orphan_img_ids = Counter(ann["image_id"] for ann in orphan_anns)
        print(f"\n  {yellow('Image IDs referenced but not in images list:')}")
        for img_id, cnt in sorted(orphan_img_ids.items()):
            print(f"    {red(f'Image ID {img_id:>6}')}  ->  {yellow(str(cnt))} annotations")
    else:
        print(f"  {ok('All annotations reference valid image IDs.')}")

    return orphan_anns


# ─────────────────────────────────────────────
#  ANALYSIS 5 — DISK vs JSON MISMATCH
# ─────────────────────────────────────────────

def analyze_disk_vs_json(images, disk_files):
    section("5 . DISK <-> JSON FILE MISMATCH")

    json_filenames   = {Path(img["file_name"]).name: img_id for img_id, img in images.items()}
    in_json_not_disk = {name for name in json_filenames if name not in disk_files}
    in_disk_not_json = {name for name in disk_files  if name not in json_filenames}

    print(f"\n  Files in JSON  : {bold(str(len(json_filenames)))}")
    print(f"  Files on disk  : {bold(str(len(disk_files)))}")
    print()

    c1 = red(bold(str(len(in_json_not_disk)))) if in_json_not_disk else green(bold("0"))
    print(f"  In JSON but NOT on disk  : {c1}")
    if in_json_not_disk:
        for name in sorted(in_json_not_disk)[:20]:
            print(f"    {red('[X]')} {name}")
        if len(in_json_not_disk) > 20:
            print(gray(f"    ... and {len(in_json_not_disk) - 20} more"))
    else:
        print(f"  {ok('All JSON-declared images are present on disk.')}")

    print()
    c2 = yellow(bold(str(len(in_disk_not_json)))) if in_disk_not_json else green(bold("0"))
    print(f"  On disk but NOT in JSON  : {c2}")
    if in_disk_not_json:
        for name in sorted(in_disk_not_json)[:20]:
            print(f"    {yellow('[!]')} {name}")
        if len(in_disk_not_json) > 20:
            print(gray(f"    ... and {len(in_disk_not_json) - 20} more"))
    else:
        print(f"  {ok('All disk images are referenced in the JSON.')}")

    return in_json_not_disk, in_disk_not_json


# ─────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────

def plot_chart(rows, output_path="coco_class_distribution.png"):
    if not HAS_MATPLOTLIB or not rows:
        return
    rows_sorted = sorted(rows, key=lambda x: -x[2])
    names  = [r[1] for r in rows_sorted]
    counts = [r[2] for r in rows_sorted]
    mean   = np.mean(counts)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 6))
    colors  = ["#e74c3c" if c < mean * 0.3 else "#f39c12" if c > mean * 3 else "#2980b9"
               for c in counts]
    ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(mean, color="#2ecc71", linewidth=1.5, linestyle="--")
    ax.set_title("COCO Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Annotations")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Under-represented (< 30% mean)"),
        mpatches.Patch(color="#f39c12", label="Over-represented (> 3x mean)"),
        mpatches.Patch(color="#2980b9", label="Balanced"),
        plt.Line2D([0],[0], color="#2ecc71", linewidth=1.5,
                   linestyle="--", label=f"Mean ({mean:.0f})"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n  {info(f'Chart saved -> {output_path}')}")
    plt.close()


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────

def print_summary(rows, unannotated_ids, orphan_anns,
                  in_json_not_disk, in_disk_not_json, ratio):
    section("SUMMARY")

    issues = []
    if ratio and ratio >= 10:
        issues.append(red(bold(f"[!] Severe class imbalance (ratio {ratio:.1f}x)")))
    elif ratio and ratio >= 3:
        issues.append(yellow(bold(f"[!] Moderate class imbalance (ratio {ratio:.1f}x)")))
    if unannotated_ids:
        issues.append(red(bold(f"[X] {len(unannotated_ids)} image(s) have NO annotations")))
    if orphan_anns:
        issues.append(red(bold(f"[X] {len(orphan_anns)} orphan annotation(s)")))
    if in_json_not_disk:
        issues.append(red(bold(f"[X] {len(in_json_not_disk)} file(s) in JSON missing from disk")))
    if in_disk_not_json:
        issues.append(yellow(bold(f"[!] {len(in_disk_not_json)} file(s) on disk not in JSON")))

    if issues:
        print()
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n  {ok('Dataset looks clean -- no issues found.')}")

    separator("=")
    print()
    return bool(issues)


# ─────────────────────────────────────────────
#  INTERACTIVE CLEANUP
# ─────────────────────────────────────────────

def _cleanup_and_return_path(raw_coco, annotations_path,
                             zero_classes, low_classes,
                             unannotated_ids, orphan_anns,
                             in_json_not_disk):
    """Wrapper that calls cleanup() and returns the saved file path (or None)."""
    return cleanup(raw_coco, annotations_path, zero_classes, low_classes,
                   unannotated_ids, orphan_anns, in_json_not_disk)


def cleanup(raw_coco, annotations_path,
            zero_classes, low_classes,
            unannotated_ids, orphan_anns,
            in_json_not_disk):

    section("CLEANUP")
    print(f"  {info('The original JSON will never be modified.')}")
    print(f"  {info('All changes are written to a NEW file.')}\n")

    cleaned = copy.deepcopy(raw_coco)
    categories_to_remove = set()
    images_to_remove     = set()
    any_action           = False

    # 1. Empty categories
    if zero_classes:
        print(f"\n  {red(bold(f'[X] Found {len(zero_classes)} category/ies with ZERO annotations:'))}")
        for cat_id, name in zero_classes:
            print(f"    {red('*')} {red(bold(f'[{cat_id}]'))} {red(name)}")
        choice = ask(
            "What do you want to do with empty categories?",
            [("r", "Remove them from the exported JSON"),
             ("i", "Ignore -- keep them as-is")]
        )
        if choice == "r":
            for cat_id, _ in zero_classes:
                categories_to_remove.add(cat_id)
            print(f"  {ok(f'Marked {len(zero_classes)} empty category/ies for removal.')}")
            any_action = True
        else:
            print(f"  {gray('-> Keeping empty categories.')}")

    # 2. Under-represented categories
    truly_low = [
        (cid, name, cnt) for cid, name, cnt in low_classes
        if cid not in categories_to_remove and cnt > 0
    ]
    if truly_low:
        print(f"\n  {yellow(bold(f'[!] Found {len(truly_low)} under-represented category/ies (< 30% of mean):'))}")
        for cat_id, name, cnt in truly_low:
            print(f"    {yellow('*')} {yellow(bold(f'[{cat_id}]'))} {yellow(name)}  {gray(f'({cnt} annotations)')}")
        choice = ask(
            "What do you want to do with under-represented categories?",
            [("r", "Remove them AND their annotations from the exported JSON"),
             ("i", "Ignore -- keep them as-is"),
             ("s", "Decide one by one")]
        )
        if choice == "r":
            for cat_id, _, _ in truly_low:
                categories_to_remove.add(cat_id)
            print(f"  {ok(f'Marked {len(truly_low)} category/ies for removal.')}")
            any_action = True
        elif choice == "s":
            for cat_id, name, cnt in truly_low:
                sub = ask(
                    f"[{cat_id}] '{yellow(name)}' ({cnt} annotations) -- remove it?",
                    [("r", red("Remove")), ("i", "Keep")]
                )
                if sub == "r":
                    categories_to_remove.add(cat_id)
                    any_action = True
        else:
            print(f"  {gray('-> Keeping under-represented categories.')}")

    # 3. Images without annotations
    if unannotated_ids:
        print(f"\n  {yellow(bold(f'[!] Found {len(unannotated_ids)} image(s) with no annotations.'))}")
        choice = ask(
            "What do you want to do with unannotated images?",
            [("r", "Remove their entries from the exported JSON"),
             ("i", "Ignore -- keep them as-is")]
        )
        if choice == "r":
            images_to_remove.update(unannotated_ids)
            print(f"  {ok(f'Marked {len(unannotated_ids)} image(s) for removal.')}")
            any_action = True
        else:
            print(f"  {gray('-> Keeping unannotated image entries.')}")

    # 4. Orphan annotations
    if orphan_anns:
        print(f"\n  {red(bold(f'[X] Found {len(orphan_anns)} annotation(s) referencing non-existent image IDs.'))}")
        choice = ask(
            "What do you want to do with orphan annotations?",
            [("r", "Remove them from the exported JSON"),
             ("i", "Ignore -- keep them as-is")]
        )
        if choice == "r":
            orphan_ann_ids = {ann["id"] for ann in orphan_anns}
            cleaned["annotations"] = [
                a for a in cleaned["annotations"] if a["id"] not in orphan_ann_ids
            ]
            print(f"  {ok(f'Removed {len(orphan_ann_ids)} orphan annotation(s).')}")
            any_action = True
        else:
            print(f"  {gray('-> Keeping orphan annotations.')}")

    # 5. JSON entries missing from disk
    if in_json_not_disk:
        print(f"\n  {red(bold(f'[X] Found {len(in_json_not_disk)} image(s) in JSON but missing from disk.'))}")
        choice = ask(
            "What do you want to do with these missing-file entries?",
            [("r", "Remove their JSON entries and annotations"),
             ("i", "Ignore -- keep them as-is")]
        )
        if choice == "r":
            img_lookup  = {img["id"]: img for img in cleaned["images"]}
            missing_ids = {
                img_id for img_id, img in img_lookup.items()
                if Path(img["file_name"]).name in in_json_not_disk
            }
            images_to_remove.update(missing_ids)
            print(f"  {ok(f'Marked {len(missing_ids)} image(s) for removal.')}")
            any_action = True
        else:
            print(f"  {gray('-> Keeping missing-file entries.')}")

    # Apply removals
    if categories_to_remove:
        before_cats = len(cleaned["categories"])
        before_anns = len(cleaned["annotations"])
        cleaned["categories"]  = [c for c in cleaned["categories"]
                                   if c["id"] not in categories_to_remove]
        cleaned["annotations"] = [a for a in cleaned["annotations"]
                                   if a["category_id"] not in categories_to_remove]
        removed_cats = before_cats - len(cleaned["categories"])
        removed_anns = before_anns - len(cleaned["annotations"])
        print(f"\n  {cyan(f'Removed {removed_cats} category/ies and {removed_anns} related annotation(s) from the exported JSON.')}")

    if images_to_remove:
        before_imgs = len(cleaned["images"])
        before_anns = len(cleaned["annotations"])
        cleaned["images"]      = [img for img in cleaned["images"]
                                   if img["id"] not in images_to_remove]
        cleaned["annotations"] = [a for a in cleaned["annotations"]
                                   if a["image_id"] not in images_to_remove]
        removed_imgs = before_imgs - len(cleaned["images"])
        removed_anns = before_anns - len(cleaned["annotations"])
        print(f"  {cyan(f'Removed {removed_imgs} image(s) and {removed_anns} related annotation(s) from the exported JSON.')}")

    if not any_action:
        print(f"\n  {gray('No changes selected -- nothing to export.')}")
        return

    print(f"\n  {bold('Result after cleanup:')}")
    print(f"    Images      : {green(bold(str(len(cleaned['images']))))}")
    print(f"    Categories  : {green(bold(str(len(cleaned['categories']))))}")
    print(f"    Annotations : {green(bold(str(len(cleaned['annotations']))))}")
    separator()

    if ask_yes_no("Export the cleaned dataset as a new JSON file?"):
        src  = Path(annotations_path)
        dest = src.parent / (src.stem + "_cleaned.json")
        print(f"\n  Default output path: {cyan(str(dest))}")
        override = input(f"  {gray('Press ENTER to accept, or type a new path: ')}").strip()
        if override:
            dest = Path(override)
        with open(dest, "w") as f:
            json.dump(cleaned, f)
        print(f"\n  {ok(f'Cleaned JSON saved -> {dest}')}")
        print(f"    {gray('Original')} : {gray(str(annotations_path))}")
        print(f"    {green('Cleaned')}  : {green(str(dest))}")
        return str(dest)
    else:
        print(f"\n  {gray('-> Export cancelled. No files were written.')}")
        return None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  AUG FACTOR COMPUTATION
# ─────────────────────────────────────────────

def compute_aug_factors(rows, max_aug=10):
    """
    Compute per-category augmentation factor based on inverse frequency
    relative to the median non-zero class count.

    Formula: aug_factor = clip(round(median / count), 1, max_aug)
    Zero-count classes get aug_factor = 0 (they should be removed).

    Returns a dict: { cat_id: aug_factor }
    """
    counts   = np.array([r[2] for r in rows], dtype=float)
    nonzero  = counts[counts > 0]
    median   = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    factors  = {}
    for cat_id, name, count in rows:
        if count == 0:
            factors[cat_id] = 0
        else:
            raw    = median / count
            factor = int(np.clip(round(raw), 1, max_aug))
            factors[cat_id] = factor
    return factors, median


def print_aug_factors(rows, factors, median):
    section("AUG FACTOR TABLE  (computed from imbalance)")
    print(f"\n  Reference median : {bold(str(int(median)))} annotations")
    print(f"  Formula          : aug_factor = clip(round(median / count), 1, 10)\n")

    h_name   = gray(pad("Category",   W_NAME,  "<"))
    h_count  = gray(pad("Count",      W_COUNT, ">"))
    h_factor = gray(pad("aug_factor", 10,      ">"))
    h_note   = gray("Note")
    print(f"  {h_name}  {h_count}  {h_factor}  {h_note}")
    separator()

    for cat_id, name, count in sorted(rows, key=lambda x: x[2]):
        f       = factors[cat_id]
        c_name  = pad(name,       W_NAME,  "<")
        c_count = pad(str(count), W_COUNT, ">")
        c_f     = pad(str(f),     10,      ">")

        if f == 0:
            note = red("[REMOVED — 0 annotations]")
            print(f"  {red(c_name)}  {red(c_count)}  {red(c_f)}  {note}")
        elif f >= 6:
            note = yellow("[!]  heavily oversampled — rare class")
            print(f"  {yellow(c_name)}  {yellow(c_count)}  {yellow(bold(c_f))}  {note}")
        elif f == 1:
            note = magenta("[^]  dominant class — no extra copies")
            print(f"  {magenta(c_name)}  {magenta(c_count)}  {magenta(c_f)}  {note}")
        else:
            note = green("[OK] moderate oversampling")
            print(f"  {c_name}  {c_count}  {green(bold(c_f))}  {note}")


def export_results(annotations_path, images_dir, output_dir,
                   rows, factors, median, ratio,
                   cleaned_annotations_path):
    """
    Write analysis_results.json inside output_dir.
    Includes pre-computed split_dir so train_unet.py can find the data directly.
    """
    output_dir   = Path(output_dir)
    images_dir_p = Path(images_dir)
    split_dir    = output_dir / (images_dir_p.name + "_split_augmented")
    out_path     = output_dir / "analysis_results.json"

    categories_data = {}
    for cat_id, name, count in rows:
        categories_data[str(cat_id)] = {
            "name":       name,
            "count":      count,
            "aug_factor": factors[cat_id],
            "remove":     factors[cat_id] == 0,
        }

    result = {
        "source_annotations":   str(Path(annotations_path).resolve()),
        "source_images":        str(images_dir_p.resolve()),
        "output_dir":           str(output_dir.resolve()),
        "split_dir":            str(split_dir.resolve()),
        "cleaned_annotations":  str(Path(cleaned_annotations_path).resolve())
                                if cleaned_annotations_path else None,
        "imbalance_ratio":      round(float(ratio), 2) if ratio else None,
        "median_annotations":   int(median),
        "removed_category_ids": [cat_id for cat_id, name, count in rows
                                 if factors[cat_id] == 0],
        "categories":           categories_data,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    section("ANALYSIS RESULTS EXPORTED")
    _msg1 = f"Saved -> {out_path}"
    _msg2 = "Next step:"
    _cmd1 = cyan("python split_and_augment.py")
    print(f"\n  {ok(_msg1)}")
    print(f"  {info(_msg2)}")
    print(f"\n    {_cmd1}")
    print()

    return str(out_path)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    global USE_COLOR

    # Read dataset_info.json from the same folder as this script
    _here              = Path(__file__).parent.resolve()
    output_dir = _here / "outputs" 
    dataset_info_path = output_dir / "dataset_info.json"

    if not dataset_info_path.exists():
        print(f"\n[X] dataset_info.json not found at: {dataset_info_path}")
        print("    Run prepare_dataset.py first.")
        raise SystemExit(1)

    print(f"\n[INFO] Reading dataset info: {dataset_info_path}")
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    annotations_path = dataset_info["annotations"]
    images_dir       = dataset_info["images_dir"]

    print(f"       Annotations  : {cyan(annotations_path)}")
    print(f"       Images dir   : {cyan(images_dir)}")

    images, categories, annotations, raw_coco = load_coco(annotations_path)
    disk_files = scan_image_folder(images_dir)

    rows, cat_counts          = analyze_classes(annotations, categories)
    ratio, zero_cls, low_cls  = analyze_imbalance(rows)
    unannotated_ids           = analyze_unannotated_images(images, annotations, disk_files)
    orphan_anns               = analyze_orphan_annotations(images, annotations)
    in_json_not_disk, in_disk_not_json = analyze_disk_vs_json(images, disk_files)

    factors, median = compute_aug_factors(rows)

    has_issues = print_summary(
        rows, unannotated_ids, orphan_anns,
        in_json_not_disk, in_disk_not_json, ratio
    )

    cleaned_path = None

    if not has_issues:
        print(f"  {ok('Dataset is clean -- no cleanup needed.')}\n")
    elif ask_yes_no("Do you want to run the cleanup to fix these issues?"):
        cleaned_path = _cleanup_and_return_path(
            raw_coco, annotations_path,
            zero_cls, low_cls,
            unannotated_ids, orphan_anns,
            in_json_not_disk,
        )
    else:
        print(f"\n  {gray('Cleanup skipped.')}\n")

    export_results(annotations_path, images_dir, output_dir,
                   rows, factors, median, ratio, cleaned_path)


if __name__ == "__main__":
    main()