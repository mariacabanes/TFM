"""
run_all.py — Master Pipeline
===================================
Runs the full dental implant analysis pipeline in order.
Steps already completed are automatically skipped.

Steps:
  1. prepare_dataset.py      — merge CVAT zips into a single COCO dataset
  2. dataset_analysis.py     — stats, cleanup, aug factors
  3. split.py                — 70 / 15 / 15 split
  4. keypoints_extraction.py — extract features + cropped images
  5. augmentation.py         — crop aug (geometric) + raw image aug (intensity)
  6. train.py                — UNet + ImplantClassifier

Usage:
    python "(0) run_all.py"                    # run all pending steps
"""

import sys
import os
import time
import argparse
import subprocess
import datetime
from pathlib import Path

_here      = Path(__file__).parent.resolve()
OUTPUT_DIR = _here / "outputs"
SPLIT_DIR  = OUTPUT_DIR / "dataset_split"
PYTHON     = sys.executable

USE_COLOR = sys.stdout.isatty()

class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED   = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    CYAN  = "\033[96m"; GRAY  = "\033[90m"; MAGENTA= "\033[95m"

def c(code, text): return f"{code}{text}{C.RESET}" if USE_COLOR else text
def bold(t):    return c(C.BOLD,    t)
def red(t):     return c(C.RED,     t)
def green(t):   return c(C.GREEN,   t)
def yellow(t):  return c(C.YELLOW,  t)
def cyan(t):    return c(C.CYAN,    t)
def gray(t):    return c(C.GRAY,    t)

def separator(char="─", width=70):
    print(gray(char * width))


# ─────────────────────────────────────────────
#  DONE CHECKS
# ─────────────────────────────────────────────

def _done_prepare():
    ann  = OUTPUT_DIR / "dataset" / "annotations.json"
    imgs = OUTPUT_DIR / "dataset" / "images"
    return ann.exists() and imgs.exists() and any(imgs.iterdir())

def _done_analysis():
    return (OUTPUT_DIR / "analysis_results.json").exists()

def _done_split():
    for split in ("train", "val", "test"):
        if not (SPLIT_DIR / split / "annotations.json").exists():
            return False
    return True

def _done_keypoints():
    for split in ("train", "val"):
        if not (SPLIT_DIR / split / "features.csv").exists():
            return False
    return True

def _done_augmentation():
    return (SPLIT_DIR / "train" / "features_augmented.csv").exists()

def _done_training():
    models_dir = OUTPUT_DIR / "models"
    if not models_dir.exists():
        return False
    return bool(list(models_dir.rglob("*.pth")) + list(models_dir.rglob("*.pt")))

def _list_existing_models():
    models_dir = OUTPUT_DIR / "models"
    if not models_dir.exists():
        return []
    files = list(models_dir.rglob("*.pth")) + list(models_dir.rglob("*.pt"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


STEPS = [
    {
        "n": 1, "script": "(1) prepare_dataset.py",
        "label": "Prepare dataset  (merge CVAT zips)",
        "done_fn": _done_prepare,
        "note": "outputs/dataset/annotations.json + images/ already exist",
        "interactive": False,
        "retrain_step": False,
    },
    {
        "n": 2, "script": "(2) dataset_analysis.py",
        "label": "Dataset analysis  (stats + cleanup + aug factors)",
        "done_fn": _done_analysis,
        "note": "outputs/analysis_results.json already exists",
        "interactive": True,
        "retrain_step": False,
    },
    {
        "n": 3, "script": "(3) split.py",
        "label": "Split dataset  (70 / 15 / 15)",
        "done_fn": _done_split,
        "note": "outputs/dataset_split/{train,val,test}/ already exist",
        "interactive": False,
        "retrain_step": False,
    },
    {
        "n": 4, "script": "(4) keypoints_extraction.py",
        "label": "Keypoints extraction  (features.csv + cropped images)",
        "done_fn": _done_keypoints,
        "note": "features.csv already exists in train/ and val/",
        "interactive": False,
        "retrain_step": False,
    },
    {
        "n": 5, "script": "(5) augmentation.py",
        "label": "Augmentation  (geometric crops + intensity raw images)",
        "done_fn": _done_augmentation,
        "note": "features_augmented.csv already exists",
        "interactive": False,
        "retrain_step": False,
    },
    {
        "n": 6, "script": "(6) train.py",
        "label": "Training  (UNet + ImplantClassifier)",
        "done_fn": _done_training,
        "note": "Model files already exist in outputs/models/",
        "interactive": False,
        "retrain_step": True,   # <-- special: always ask / --retrain flag
    },
]


# ─────────────────────────────────────────────
#  RETRAIN PROMPT
# ─────────────────────────────────────────────

def _ask_retrain() -> bool:
    """
    Show existing models and ask the user whether to retrain.
    Returns True if the user wants to retrain.
    """
    models = _list_existing_models()
    print(f"\n  {yellow('[!]')} Training has already been run.")
    print(f"       Existing model files ({len(models)}):")
    for m in models[:5]:
        mtime = datetime.datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"         {gray(mtime)}  {m.name}")
    if len(models) > 5:
        print(f"         ... and {len(models) - 5} more")

    print()
    print(f"  {bold('Do you want to retrain? (y/N)')}", end=" ", flush=True)
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in ("y", "yes", "s", "si", "sí")


# ─────────────────────────────────────────────
#  RUN A STEP
# ─────────────────────────────────────────────

def run_step(step: dict) -> bool:
    script = _here / step["script"]
    if not script.exists():
        print(f"  {red('[X]')} Script not found: {script}")
        return False

    cmd = [PYTHON, str(script)]
    print(f"\n  {cyan('$')} {' '.join(str(x) for x in cmd)}")
    separator()
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=str(_here))
    elapsed = time.time() - t0
    separator()
    if ret.returncode == 0:
        print(f"  {green('[OK]')} Finished in {elapsed:.1f}s")
        return True
    else:
        print(f"  {red('[FAIL]')} Exit code {ret.returncode}  ({elapsed:.1f}s)")
        return False


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Master pipeline")
    parser.add_argument("--from",    type=int, dest="from_step",  default=1,    metavar="N")
    parser.add_argument("--only",    type=int, dest="only_step",   default=None, metavar="N")
    parser.add_argument("--skip",    type=int, dest="skip_steps",  default=[],  nargs="+", metavar="N")
    parser.add_argument("--force",   action="store_true", help="Re-run all steps including completed ones")
    parser.add_argument("--retrain", action="store_true", help="Re-run training even if models exist (no prompt)")
    args = parser.parse_args()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print(bold("█" * 70))
    print(bold("  DENTAL IMPLANT ANALYSIS — FULL PIPELINE"))
    print(f"  Started  : {now}")
    print(f"  Python   : {PYTHON}")
    print(f"  Root dir : {_here}")
    print(bold("█" * 70))

    if args.only_step is not None:
        steps_to_run = [s for s in STEPS if s["n"] == args.only_step]
    else:
        steps_to_run = [s for s in STEPS
                        if s["n"] >= args.from_step
                        and s["n"] not in args.skip_steps]

    if not steps_to_run:
        print(f"\n{yellow('[!]')} No steps selected.")
        return

    # ── Status table ─────────────────────────────────────────────────
    print(f"\n  {'Step':<6} {'Script':<38} Status")
    separator()
    for s in STEPS:
        in_run = s in steps_to_run
        if not in_run:
            status = gray("SKIP (excluded by flags)")
        elif args.force:
            status = yellow("FORCE RUN")
        elif s["retrain_step"] and args.retrain:
            status = yellow("RETRAIN")
        elif s["done_fn"]():
            if s["retrain_step"]:
                status = green("DONE  (will ask to retrain)")
            else:
                status = green("DONE  (will skip)")
        else:
            status = cyan("PENDING")
        print(f"  [{s['n']}]   {s['script']:<38} {status}")
    separator()

    interactive = [s for s in steps_to_run
                   if s["interactive"] and (args.force or not s["done_fn"]())]
    if interactive:
        print(f"\n{yellow('[!]')} Step 2 (dataset_analysis) may prompt for input during cleanup.")

    # ── Execute ──────────────────────────────────────────────────────
    results = {}
    t_start = time.time()

    for step in steps_to_run:
        n = step["n"]
        print(f"\n{'█'*70}")
        print(f"  STEP {n} / {len(STEPS)}  —  {bold(step['label'])}")
        print(f"{'█'*70}")

        already_done = step["done_fn"]()

        # ── Training step special logic ───────────────────────────────
        if step["retrain_step"] and already_done and not args.force:
            if args.retrain:
                # --retrain flag: skip prompt, just run
                print(f"\n  {yellow('[RETRAIN]')} --retrain flag set, running training again.")
            else:
                # Ask the user
                want_retrain = _ask_retrain()
                if not want_retrain:
                    print(f"\n  {gray('[SKIP]')}  Training skipped by user.")
                    results[n] = "skipped"
                    continue

        # ── All other steps ───────────────────────────────────────────
        elif already_done and not args.force and not step["retrain_step"]:
            print(f"\n  {green('[SKIP]')}  {step['note']}")
            results[n] = "skipped"
            continue

        ok = run_step(step)
        results[n] = "ok" if ok else "failed"

        if not ok:
            print(f"\n{red('█'*70)}")
            print(f"  {red('[PIPELINE STOPPED]')}  Step {n} failed.")
            print(f"  Fix the issue and re-run with:")
            print(f"    {cyan(f'python run_all.py --from {n}')}")
            if step["retrain_step"]:
                print(f"  Or to retrain directly:")
                print(f"    {cyan('python run_all.py --only 6 --retrain')}")
            print(f"{red('█'*70)}")
            break

    # ── Summary ──────────────────────────────────────────────────────
    total = time.time() - t_start
    print(f"\n{'█'*70}")
    print(bold("  PIPELINE SUMMARY"))
    print(f"{'█'*70}")
    separator()
    for s in STEPS:
        n   = s["n"]
        res = results.get(n, "not reached")
        if res == "ok":        tag = green("[OK]     ")
        elif res == "skipped": tag = gray("[SKIP]   ")
        elif res == "failed":  tag = red("[FAIL]   ")
        else:                  tag = gray("[-------]")
        print(f"  {tag} [{n}] {s['label']}")
    separator()

    failed = [n for n, r in results.items() if r == "failed"]
    if failed:
        print(f"\n  {red(f'Pipeline failed at step {failed[0]}.')}")
        print(f"  Re-run from that step:  {cyan(f'python run_all.py --from {failed[0]}')}")
    else:
        done    = sum(1 for r in results.values() if r == "ok")
        skipped = sum(1 for r in results.values() if r == "skipped")
        print(f"\n  {green('All selected steps completed successfully.')}")
        print(f"  Steps run: {done}   skipped: {skipped}")

    m, s = divmod(int(total), 60)
    print(f"  Total time: {m}m {s}s")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()