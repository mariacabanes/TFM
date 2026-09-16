"""
(10) tcav_analysis.py — TCAV (Testing with Concept Activation Vectors) for the image
branch of the trained ImplantClassifier (EfficientNetB2 + GNN + scalar features).

Implements Kim et al. 2018 ("Interpretability Beyond Feature Attribution: Quantitative
Testing with Concept Activation Vectors") against the clinical concepts persisted by
step 4 (`(4) keypoints_extraction.py` -> `build_concept_dataset()`):

    - tipo_implante_U / tipo_implante_Recto   (implant shape)
    - steep_thread_angle                      (pronounced thread angle vs shallow)

For each concept:
    1. A linear Concept Activation Vector (CAV) is trained on the EfficientNetB2
       pooled features (the `cnn_backbone` output, 1408-d, right before `cnn_fc`)
       using the `concepts/<name>/{positive,negative}` probe images built in step 4.
    2. The directional derivative of each brand/diameter logit w.r.t. that layer is
       projected onto the CAV, for every test example of the class being explained.
    3. The TCAV score (fraction of positive directional derivatives) is bootstrapped
       over several CAV refits and compared, per class, against a null distribution
       built from CAVs trained on random (non-concept) image splits, via a two-sided
       Welch t-test — the statistical significance test from the original paper.

This script only *reads* the checkpoint trained by `fed_manual 1`'s `(6) train.py`;
it never updates the model in place. Concept probe images and the test split used to
compute directional derivatives come from this folder's own `outputs/dataset_split/`
(built by this folder's step 1-4 scripts).

Usage:
    python "(10) tcav_analysis.py"
    python "(10) tcav_analysis.py" --split test --concepts-split train --n-bootstrap 10
"""

import argparse
import ast
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch_geometric.utils import dense_to_sparse

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

SEED = 1225
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SCALAR_COLS = [
    "distance_cm", "pixel_per_cm", "bbox_width", "bbox_height",
    "implant_bbox_ratio", "angle_left_valley", "angle_right_valley",
]
NODE_KEYS = [
    "left_top", "right_top", "left_bottom", "right_bottom",
    "interior_left", "interior_right",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_here = Path(__file__).parent.resolve()


# =============================================================================
#  PATH DISCOVERY
# =============================================================================

def find_fed1_dir() -> Path:
    """Locates the sibling `fed_manual 1` folder (trained models + train CSV)."""
    env = os.environ.get("FED_MANUAL_1_DIR")
    if env:
        return Path(env).resolve()
    repo_root = _here.parent
    candidates = sorted(p for p in repo_root.glob("fed_manual 1*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(
            "Could not find a 'fed_manual 1*' folder next to 'fed_manual 2'. "
            "Set FED_MANUAL_1_DIR to override."
        )
    return candidates[0]


def find_latest_checkpoint(models_dir: Path) -> Path:
    candidates = list(models_dir.rglob("EfficientNetGNN_best_valDiam_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No ImplantClassifier checkpoint found under {models_dir}. "
            "Run fed_manual 1's '(6) train.py' first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# =============================================================================
#  LABEL MAPS / SCALER  (must mirror fed_manual 1's train.py exactly)
# =============================================================================

def build_label_maps_and_scaler(train_csv: Path):
    df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(df[SCALAR_COLS])
    brands = sorted(df["brand"].unique())
    diameters = sorted(df["implant_diameter_mm"].unique())
    brand_map = {b: i for i, b in enumerate(brands)}
    diameter_map = {d: i for i, d in enumerate(diameters)}
    return brand_map, diameter_map, scaler


def _parse_node(val):
    s = str(val).strip()
    s = re.sub(r"\bnan\b", "0.0", s, flags=re.IGNORECASE)
    s = re.sub(r"\binf\b", "0.0", s, flags=re.IGNORECASE)
    try:
        r = ast.literal_eval(s)
        if isinstance(r, tuple):
            return [float(r[0]), float(r[1])]
        return [float(r), 0.0]
    except (ValueError, SyntaxError):
        return [0.0, 0.0]


def build_edge_index(n_nodes: int, batch_size: int, device) -> torch.Tensor:
    adj = torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)
    ei, _ = dense_to_sparse(adj)
    return torch.cat([ei + b * n_nodes for b in range(batch_size)], dim=1).to(device)


# =============================================================================
#  IMAGE LOADING  (mirrors the mask-multiplied preprocessing used at train time)
# =============================================================================

def load_masked_image(image_path: Path, mask_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        image_np = np.array(image)
        mask_np = np.array(mask)
        image = Image.fromarray((image_np * (mask_np > 0)[..., None]).astype(np.uint8))
    return image


def mask_path_for(image_path: Path, split_dir: Path) -> Path:
    return split_dir / "masks" / f"{image_path.stem}_seg.png"


# =============================================================================
#  PROBE DATASET  (test examples used to compute directional derivatives)
# =============================================================================

class TCAVProbeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, mask_dir: Path):
        self.data = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / f"{row['image_name']}.png"
        mask_path = self.mask_dir / f"{row['image_name']}_seg.png"
        image = IMAGE_TRANSFORM(load_masked_image(img_path, mask_path))
        node_feats = torch.tensor([_parse_node(row[k]) for k in NODE_KEYS], dtype=torch.float32)
        scalar_feat = torch.tensor([row[c] for c in SCALAR_COLS], dtype=torch.float32)
        return image, node_feats, scalar_feat, int(row["brand_idx"]), int(row["diameter_idx"])


# =============================================================================
#  ACTIVATIONS / CAV
# =============================================================================

class ActivationGrabber:
    """Hooks `module` and retains the gradient of its output for backprop."""

    def __init__(self, module: torch.nn.Module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out.retain_grad()
        self.activation = out

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def extract_image_activations(model, image_paths: List[Path], split_dir: Path,
                               device, batch_size: int = 16) -> np.ndarray:
    """Runs the CNN branch only (no graph/scalar branch needed) to get bottleneck features."""
    grabber = ActivationGrabber(model.cnn_backbone)
    feats = []
    model.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = [IMAGE_TRANSFORM(load_masked_image(p, mask_path_for(p, split_dir))) for p in batch_paths]
        batch = torch.stack(imgs).to(device)
        model.cnn_backbone(batch)
        feats.append(grabber.activation.detach().cpu().numpy())
    grabber.remove()
    return np.concatenate(feats, axis=0)


def train_cav(pos_acts: np.ndarray, neg_acts: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, float]:
    """Linear CAV: normal vector of a logistic-regression boundary pos-vs-neg activations."""
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    v = clf.coef_[0]
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype(np.float32), float(acc)


def directional_derivatives(model, loader: DataLoader, cav_vector: np.ndarray,
                             head: str, class_idx: int, device) -> np.ndarray:
    """S_C(x) = grad(logit_class w.r.t. bottleneck activation) . cav_vector, per example."""
    grabber = ActivationGrabber(model.cnn_backbone)
    cav_t = torch.tensor(cav_vector, device=device)
    scores = []
    model.eval()
    for images, node_feats, scalar_feat, brand_idx, diameter_idx in loader:
        images = images.to(device)
        node_feats = node_feats.to(device)
        scalar_feat = scalar_feat.to(device)
        B = images.shape[0]
        batch_vec = torch.arange(B, device=device).repeat_interleave(6)
        edge_index = build_edge_index(6, B, device)

        model.zero_grad(set_to_none=True)
        out_b, out_d = model(images, node_feats.view(-1, 2), edge_index, batch_vec, scalar_feat)
        logits = out_b if head == "brand" else out_d
        target = logits[:, class_idx].sum()
        target.backward()

        grad = grabber.activation.grad  # (B, 1408)
        s = (grad * cav_t).sum(dim=1)
        scores.extend(s.detach().cpu().tolist())
    grabber.remove()
    return np.array(scores)


def tcav_score(directional_scores: np.ndarray) -> float:
    return float(np.mean(directional_scores > 0))


# =============================================================================
#  CONCEPT PROBE SETS + BOOTSTRAP / SIGNIFICANCE TESTING
# =============================================================================

def list_concept_images(concept_dir: Path, max_per_class: int, rng: random.Random):
    pos = sorted((concept_dir / "positive").glob("*.png"))
    neg = sorted((concept_dir / "negative").glob("*.png"))
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:max_per_class], neg[:max_per_class]


def bootstrap_concept_tcav(model, pos_paths, neg_paths, split_dir, probe_loader,
                            head, class_idx, device, n_boot, subsample_frac, rng):
    scores, accs = [], []
    n_pos = max(2, int(len(pos_paths) * subsample_frac))
    n_neg = max(2, int(len(neg_paths) * subsample_frac))
    for b in range(n_boot):
        pos_sample = rng.sample(pos_paths, min(n_pos, len(pos_paths)))
        neg_sample = rng.sample(neg_paths, min(n_neg, len(neg_paths)))
        pos_acts = extract_image_activations(model, pos_sample, split_dir, device)
        neg_acts = extract_image_activations(model, neg_sample, split_dir, device)
        cav, acc = train_cav(pos_acts, neg_acts, seed=b)
        accs.append(acc)
        S = directional_derivatives(model, probe_loader, cav, head, class_idx, device)
        scores.append(tcav_score(S))
    return np.array(scores), np.array(accs)


def random_baseline_tcav(model, image_pool: List[Path], split_dir, probe_loader,
                          head, class_idx, device, n_random, sample_size, rng):
    scores = []
    pool = image_pool[:]
    for _ in range(n_random):
        if len(pool) < 2 * sample_size:
            sample = rng.choices(pool, k=2 * sample_size)
        else:
            sample = rng.sample(pool, 2 * sample_size)
        pos_r, neg_r = sample[:sample_size], sample[sample_size:2 * sample_size]
        pos_acts = extract_image_activations(model, pos_r, split_dir, device)
        neg_acts = extract_image_activations(model, neg_r, split_dir, device)
        cav, _ = train_cav(pos_acts, neg_acts, seed=rng.randint(0, 10 ** 6))
        S = directional_derivatives(model, probe_loader, cav, head, class_idx, device)
        scores.append(tcav_score(S))
    return np.array(scores)


# =============================================================================
#  OUTPUT
# =============================================================================

def plot_tcav_results(results: Dict, out_path: Path):
    rows = []
    for concept, per_class in results.items():
        for label, r in per_class.items():
            rows.append((concept, label, r["tcav_score_mean"], r["tcav_score_std"], r["significant"]))
    if not rows:
        print("[plot] Nothing to plot (no concept/class combination had enough examples).")
        return

    df = pd.DataFrame(rows, columns=["concept", "class", "score", "std", "significant"])
    concepts = df["concept"].unique()

    fig, axes = plt.subplots(len(concepts), 1, figsize=(10, 3.5 * len(concepts)), squeeze=False)
    for ax, concept in zip(axes[:, 0], concepts):
        sub = df[df["concept"] == concept].sort_values("score", ascending=False)
        colors = ["#2a9d8f" if sig else "#b0b0b0" for sig in sub["significant"]]
        ax.barh(sub["class"], sub["score"], xerr=sub["std"], color=colors)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"TCAV scores — concept: {concept}  (green = significant vs random, p<0.05)")
        ax.set_xlabel("TCAV score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="TCAV analysis for the ImplantClassifier's image branch.")
    p.add_argument("--split", default="test", help="Split used as class-conditional probe examples.")
    p.add_argument("--concepts-split", default="train", help="Split whose concepts/ folder is used to train CAVs.")
    p.add_argument("--model", default=None, help="Path to a specific checkpoint (.pt). Default: latest in fed_manual 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-concept-images", type=int, default=60, help="Cap per concept class (positive/negative).")
    p.add_argument("--min-examples-per-class", type=int, default=5, help="Skip a (head, class) if fewer test examples exist.")
    p.add_argument("--max-examples-per-class", type=int, default=40, help="Cap on directional-derivative examples per class.")
    p.add_argument("--n-bootstrap", type=int, default=8, help="Number of CAV refits per concept (statistical robustness).")
    p.add_argument("--n-random-cavs", type=int, default=12, help="Number of random-concept CAVs for the null distribution.")
    p.add_argument("--subsample-frac", type=float, default=0.8, help="Fraction of probe images used per bootstrap CAV.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for the two-sided t-test.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(SEED)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    fed1_dir = find_fed1_dir()
    fed2_dir = _here
    print(f"[INFO] fed_manual 1 (models, read-only): {fed1_dir}")
    print(f"[INFO] fed_manual 2 (concepts + probes) : {fed2_dir}")

    sys.path.insert(0, str(fed1_dir))
    from models.ImplantClassifier import ImplantClassifier  # noqa: E402

    train_csv = fed1_dir / "outputs" / "dataset_split" / "train" / "features_augmented.csv"
    if not train_csv.exists():
        train_csv = fed1_dir / "outputs" / "dataset_split" / "train" / "features.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Could not find train features CSV under {train_csv.parent}.")
    brand_map, diameter_map, scaler = build_label_maps_and_scaler(train_csv)
    print(f"[INFO] Brands ({len(brand_map)}): {list(brand_map)}")
    print(f"[INFO] Diameters ({len(diameter_map)}): {list(diameter_map)}")

    ckpt_path = Path(args.model) if args.model else find_latest_checkpoint(fed1_dir / "outputs" / "models")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    model = ImplantClassifier(n_brands=len(brand_map), n_diameters=len(diameter_map)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ---- Probe split (class-conditional examples for directional derivatives) ----
    probe_split_dir = fed2_dir / "outputs" / "dataset_split" / args.split
    probe_csv = probe_split_dir / "features.csv"
    if not probe_csv.exists():
        raise FileNotFoundError(f"{probe_csv} not found. Run this folder's '(4) keypoints_extraction.py' first.")
    probe_df = pd.read_csv(probe_csv)

    existing = {p.stem for p in (probe_split_dir / "cropped_images").glob("*.png")}
    probe_df = probe_df[probe_df["image_name"].isin(existing)]
    probe_df = probe_df[probe_df["brand"].isin(brand_map) & probe_df["implant_diameter_mm"].isin(diameter_map)]
    probe_df = probe_df.dropna(subset=SCALAR_COLS).reset_index(drop=True)
    probe_df[SCALAR_COLS] = scaler.transform(probe_df[SCALAR_COLS])
    probe_df["brand_idx"] = probe_df["brand"].map(brand_map)
    probe_df["diameter_idx"] = probe_df["implant_diameter_mm"].map(diameter_map)
    print(f"[INFO] Probe split '{args.split}': {len(probe_df)} usable examples.")

    # ---- Concept probe sets + random-baseline pool ----
    concepts_split_dir = fed2_dir / "outputs" / "dataset_split" / args.concepts_split
    concepts_dir = concepts_split_dir / "concepts"
    if not concepts_dir.exists():
        raise FileNotFoundError(
            f"{concepts_dir} not found. Run '(4) keypoints_extraction.py' "
            f"(build_concept_dataset) for split '{args.concepts_split}' first."
        )
    concept_names = sorted(d.name for d in concepts_dir.iterdir() if d.is_dir())
    print(f"[INFO] Concepts found: {concept_names}")

    image_pool = list((concepts_split_dir / "cropped_images").glob("*.png"))

    results: Dict[str, Dict[str, dict]] = {}

    for concept in concept_names:
        concept_dir = concepts_dir / concept
        pos_paths, neg_paths = list_concept_images(concept_dir, args.max_concept_images, rng)
        if len(pos_paths) < 5 or len(neg_paths) < 5:
            print(f"[skip] concept '{concept}': not enough probe images (pos={len(pos_paths)}, neg={len(neg_paths)})")
            continue

        results[concept] = {}
        for head, label_map, idx_col in (
            ("brand", brand_map, "brand_idx"),
            ("diameter", diameter_map, "diameter_idx"),
        ):
            for cls_name, cls_idx in label_map.items():
                subset = probe_df[probe_df[idx_col] == cls_idx]
                if len(subset) < args.min_examples_per_class:
                    continue
                subset = subset.sample(min(len(subset), args.max_examples_per_class), random_state=SEED)

                probe_ds = TCAVProbeDataset(subset, probe_split_dir / "cropped_images", probe_split_dir / "masks")
                probe_loader = DataLoader(probe_ds, batch_size=args.batch_size, shuffle=False)

                concept_scores, cav_accs = bootstrap_concept_tcav(
                    model, pos_paths, neg_paths, concepts_split_dir, probe_loader,
                    head, cls_idx, device, args.n_bootstrap, args.subsample_frac, rng,
                )
                random_scores = random_baseline_tcav(
                    model, image_pool, concepts_split_dir, probe_loader,
                    head, cls_idx, device, args.n_random_cavs,
                    sample_size=min(len(pos_paths), len(neg_paths)), rng=rng,
                )

                _, pval = ttest_ind(concept_scores, random_scores, equal_var=False)
                significant = bool(pval < args.alpha)

                label = f"{head}={cls_name}"
                results[concept][label] = {
                    "head": head,
                    "class": str(cls_name),
                    "tcav_score_mean": float(concept_scores.mean()),
                    "tcav_score_std": float(concept_scores.std()),
                    "random_score_mean": float(random_scores.mean()),
                    "random_score_std": float(random_scores.std()),
                    "cav_accuracy_mean": float(cav_accs.mean()),
                    "p_value": float(pval),
                    "significant": significant,
                    "n_examples": int(len(subset)),
                }
                flag = "SIGNIF" if significant else "n.s."
                print(f"[{concept}] {label}: TCAV={concept_scores.mean():.2f}±{concept_scores.std():.2f} "
                      f"(random={random_scores.mean():.2f}, p={pval:.3f}, {flag}, "
                      f"CAV acc={cav_accs.mean():.2f}, n={len(subset)})")

    out_dir = fed2_dir / "outputs" / "explanations" / "tcav"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tcav_scores.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {out_dir / 'tcav_scores.json'}")

    plot_tcav_results(results, out_dir / "tcav_scores.png")


if __name__ == "__main__":
    main()
