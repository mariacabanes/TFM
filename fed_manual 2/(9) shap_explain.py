"""
(9) shap_explain.py — SHAP vs. permutation importance over the scalar-feature branch
(7 features) of the trained ImplantClassifier.

Two independent ways of asking "which scalar feature matters most for this prediction?":

    - SHAP (Lundberg & Lee 2017, "A Unified Approach to Interpreting Model
      Predictions", NeurIPS). Per-example Shapley values of the 7 scalar features
      (`distance_cm`, `pixel_per_cm`, `bbox_width`, `bbox_height`,
      `implant_bbox_ratio`, `angle_left_valley`, `angle_right_valley`) for the
      target class's predicted probability. With only 7 features the full set of
      2^7 = 128 coalitions is enumerable, so `shap`'s *exact* explainer is used —
      no sampling approximation (KernelSHAP) and no gradient-based approximation
      (DeepSHAP / GradientSHAP) needed. Absent features are marginalized over a
      background sample of the (standardized) training set via
      `shap.maskers.Independent` (interventional SHAP).

    - Permutation importance (Breiman 2001; Fisher et al. 2019). Global,
      model-agnostic: shuffle one scalar column across the probe split and measure
      the drop in accuracy (vs. ground truth) and in the target class's predicted
      probability, averaged over several shuffles.

Only the scalar branch is perturbed. For each example, the image and graph branches
are run once (frozen) and their embeddings (`img_feat`, `gnn_feat`) are held fixed
while SHAP/permutation vary the scalar input — mathematically identical to a full
forward per coalition, just without redoing the EfficientNetB2 pass 128 x background
times per example (same approach as step 8).

The SHAP feature values shown on plots are in the original (un-standardized) units,
recovered with the same StandardScaler that `fed_manual 1`'s `(6) train.py` fitted;
the model itself is always fed the standardized values.

This script only *reads* the checkpoint trained by `fed_manual 1`'s `(6) train.py`;
it never updates the model in place.

Requires: `pip install shap`

Usage:
    python "(9) shap_explain.py"
    python "(9) shap_explain.py" --heads brand --target true --max-examples-per-class 30
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
from scipy.stats import spearmanr, pearsonr

from common_explain import (
    SEED, HERE, SCALAR_COLS, NODE_KEYS, ImplantProbeDataset,
    build_edge_index, build_label_maps_and_scaler, find_fed1_dir, find_train_csv,
    load_model, load_probe_df,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_NODES = len(NODE_KEYS)
N_FEATS = len(SCALAR_COLS)


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


# =============================================================================
#  FROZEN IMAGE/GRAPH EMBEDDINGS + SCALAR-ONLY FORWARD
# =============================================================================

@torch.no_grad()
def precompute_embeddings(model, ds: ImplantProbeDataset, device, batch_size: int):
    """Runs the image and graph branches once per example.
    Returns img_feat (N, 128), gnn_feat (N, 64), scalar (N, 7), brand_idx (N,),
    diameter_idx (N,), image_names (list)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    img_feats, gnn_feats, scalars, b_idx, d_idx, names = [], [], [], [], [], []
    for images, node_feats, scalar_feat, brand_idx, diameter_idx, image_name in loader:
        B = images.shape[0]
        images = images.to(device)
        batch_vec = torch.arange(B, device=device).repeat_interleave(N_NODES)
        edge_index = build_edge_index(N_NODES, B, device)
        x = torch.relu(model.gcn1(node_feats.view(-1, 2).to(device), edge_index))
        x = model.gcn2(x, edge_index)
        gnn_feats.append(global_mean_pool(x, batch_vec).cpu())
        img_feats.append(model.cnn_fc(model.cnn_backbone(images)).cpu())
        scalars.append(scalar_feat)
        b_idx.append(brand_idx)
        d_idx.append(diameter_idx)
        names.extend(image_name)
    return (torch.cat(img_feats), torch.cat(gnn_feats), torch.cat(scalars).numpy(),
            torch.cat(b_idx).numpy(), torch.cat(d_idx).numpy(), names)


@torch.no_grad()
def head_probs(model, img_feat: torch.Tensor, gnn_feat: torch.Tensor, scalar: np.ndarray,
               head: str, device) -> np.ndarray:
    """Same computation as ImplantClassifier.forward from the fusion layer onward.
    img_feat/gnn_feat are (N, d) or (1, d) (broadcast against N scalar rows)."""
    s = torch.as_tensor(scalar, dtype=torch.float32, device=device)
    n = s.shape[0]
    img = img_feat.to(device).expand(n, -1) if img_feat.shape[0] == 1 else img_feat.to(device)
    gnn = gnn_feat.to(device).expand(n, -1) if gnn_feat.shape[0] == 1 else gnn_feat.to(device)
    out = model.fc(torch.cat([img, gnn, model.scalar_fc(s)], dim=1))
    logits = model.brand_head(out) if head == "brand" else model.diameter_head(out)
    return TF.softmax(logits, dim=1).cpu().numpy()


# =============================================================================
#  METHOD 1 — Exact SHAP over the 7 scalar features
# =============================================================================

def shap_explain_instance(model, img_feat_i: torch.Tensor, gnn_feat_i: torch.Tensor,
                          scalar_i: np.ndarray, masker, head: str, class_idx: int, device):
    """Returns (shap_values[7], base_value) for the target class's predicted probability."""
    def f(X):
        return head_probs(model, img_feat_i, gnn_feat_i, X, head, device)[:, class_idx]

    explainer = shap.explainers.Exact(f, masker)
    expl = explainer(scalar_i[None, :], silent=True)
    return np.asarray(expl.values[0], dtype=float), float(np.asarray(expl.base_values).reshape(-1)[0])


# =============================================================================
#  METHOD 2 — Permutation importance
# =============================================================================

def permutation_importance(model, img_feat, gnn_feat, scalar: np.ndarray, y_true: np.ndarray,
                           target_idx: np.ndarray, head: str, device, n_repeats: int, rng):
    """Returns dict with per-feature mean/std drop in accuracy and in target-class probability."""
    rows = np.arange(len(scalar))

    def score(X):
        p = head_probs(model, img_feat, gnn_feat, X, head, device)
        return (p.argmax(axis=1) == y_true).mean(), p[rows, target_idx].mean()

    base_acc, base_prob = score(scalar)
    acc_drop = np.zeros((n_repeats, N_FEATS))
    prob_drop = np.zeros((n_repeats, N_FEATS))
    for r in range(n_repeats):
        for k in range(N_FEATS):
            X = scalar.copy()
            X[:, k] = X[rng.permutation(len(X)), k]
            acc, prob = score(X)
            acc_drop[r, k] = base_acc - acc
            prob_drop[r, k] = base_prob - prob
    return {
        "base_accuracy": float(base_acc),
        "base_target_prob": float(base_prob),
        "acc_drop_mean": acc_drop.mean(axis=0), "acc_drop_std": acc_drop.std(axis=0),
        "prob_drop_mean": prob_drop.mean(axis=0), "prob_drop_std": prob_drop.std(axis=0),
    }


# =============================================================================
#  VISUALIZATION
# =============================================================================

def save_waterfall(values, base_value, data_raw, out_path: Path, title: str):
    expl = shap.Explanation(values=values, base_values=base_value, data=data_raw, feature_names=SCALAR_COLS)
    shap.plots.waterfall(expl, max_display=N_FEATS, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_beeswarm(values, base_values, data_raw, out_path: Path, title: str):
    expl = shap.Explanation(values=values, base_values=base_values, data=data_raw, feature_names=SCALAR_COLS)
    shap.plots.beeswarm(expl, max_display=N_FEATS, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_comparison_figure(shap_mean_abs, shap_std_abs, perm_mean, perm_std, out_path: Path, title: str,
                           perm_label: str):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(N_FEATS)
    width = 0.38
    # Each method normalized to sum 1 so the two scales are comparable side by side
    s_norm = shap_mean_abs.sum() + 1e-12
    p_norm = np.abs(perm_mean).sum() + 1e-12
    ax.bar(x - width / 2, shap_mean_abs / s_norm, width, yerr=shap_std_abs / s_norm,
           label="SHAP mean |value|", color="#2a9d8f", capsize=3)
    ax.bar(x + width / 2, perm_mean / p_norm, width, yerr=perm_std / p_norm,
           label=perm_label, color="#e76f51", capsize=3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(SCALAR_COLS, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative importance (normalized)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_class_heatmap(matrix: np.ndarray, class_names, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(class_names) + 2))
    vmax = np.abs(matrix).max() + 1e-12
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N_FEATS)); ax.set_xticklabels(SCALAR_COLS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels([str(c) for c in class_names], fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean SHAP value (Δ prob)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None, None, None
    rho, p = spearmanr(a, b)
    r, _ = pearsonr(a, b)
    return (float(rho) if not np.isnan(rho) else None,
            float(p) if not np.isnan(p) else None,
            float(r) if not np.isnan(r) else None)


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SHAP vs permutation importance over the 7 scalar features.")
    p.add_argument("--split", default="test", help="Split whose examples are explained.")
    p.add_argument("--heads", default="brand,diameter", help="Comma-separated subset of {brand,diameter}.")
    p.add_argument("--target", choices=["predicted", "true"], default="predicted",
                   help="Explain the model's own prediction, or the ground-truth label.")
    p.add_argument("--model", default=None, help="Path to a specific checkpoint (.pt). Default: latest in fed_manual 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for the frozen image/graph pass.")
    p.add_argument("--background-size", type=int, default=100,
                   help="Training rows used as the SHAP background distribution.")
    p.add_argument("--min-examples-per-class", type=int, default=5)
    p.add_argument("--max-examples-per-class", type=int, default=30,
                   help="Cap on SHAP-explained examples per class.")
    p.add_argument("--max-images-per-class", type=int, default=3,
                   help="Cap on per-example waterfall figures saved per class.")
    p.add_argument("--n-repeats", type=int, default=20, help="Shuffles per feature for permutation importance.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(SEED)
    print(f"[INFO] Device: {device}")

    fed1_dir = find_fed1_dir()
    fed2_dir = HERE
    print(f"[INFO] fed_manual 1 (models, read-only): {fed1_dir}")
    print(f"[INFO] fed_manual 2 (probes)            : {fed2_dir}")

    train_csv = find_train_csv(fed1_dir)
    brand_map, diameter_map, scaler = build_label_maps_and_scaler(train_csv)
    model, ckpt_path = load_model(fed1_dir, brand_map, diameter_map, device, args.model)
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[INFO] Checkpoint: {ckpt_path}")

    # ---- SHAP background: standardized training-set scalar features ----
    train_scalar = pd.read_csv(train_csv)[SCALAR_COLS].dropna()
    background = scaler.transform(train_scalar.sample(min(len(train_scalar), args.background_size),
                                                      random_state=SEED)).astype(np.float32)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    print(f"[INFO] SHAP background: {len(background)} rows from {train_csv.name}")

    # ---- Probe examples: run the frozen image/graph branches once ----
    probe_df, probe_split_dir = load_probe_df(fed2_dir, args.split, brand_map, diameter_map, scaler)
    print(f"[INFO] Probe split '{args.split}': {len(probe_df)} usable examples.")
    ds = ImplantProbeDataset(probe_df, probe_split_dir / "cropped_images", probe_split_dir / "masks")
    img_feat, gnn_feat, scalar, brand_true, diam_true, image_names = precompute_embeddings(
        model, ds, device, args.batch_size)
    scalar_raw = scaler.inverse_transform(scalar)

    heads_wanted = set(args.heads.split(","))
    out_dir = fed2_dir / "outputs" / "explanations" / "shap"
    per_image_dir = out_dir / "per_image"
    aggregate_dir = out_dir / "aggregate"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    summary = {"split": args.split, "target": args.target, "background_size": int(len(background))}

    for head, label_map, y_true in (
        ("brand", brand_map, brand_true),
        ("diameter", diameter_map, diam_true),
    ):
        if head not in heads_wanted:
            continue
        inv_map = {v: k for k, v in label_map.items()}

        probs = head_probs(model, img_feat, gnn_feat, scalar, head, device)
        y_pred = probs.argmax(axis=1)
        target_all = y_pred if args.target == "predicted" else y_true

        # ---- Permutation importance (global, on every usable probe example) ----
        perm = permutation_importance(model, img_feat, gnn_feat, scalar, y_true, target_all,
                                      head, device, args.n_repeats, rng)
        print(f"[{head}] scalar-branch-only baseline accuracy={perm['base_accuracy']:.3f}")

        # ---- SHAP per example (class-conditional, capped per class) ----
        head_summary = {"classes": {}}
        all_vals, all_base, all_rows = [], [], []
        class_rows, class_names = [], []

        for cls_idx in sorted(inv_map):
            cls_name = inv_map[cls_idx]
            # Group by the explained target (true label or prediction), like steps 7/8 group by class
            group_col = y_true if args.target == "true" else y_pred
            rows = np.where(group_col == cls_idx)[0]
            if len(rows) < args.min_examples_per_class:
                continue
            if len(rows) > args.max_examples_per_class:
                rows = rng.choice(rows, args.max_examples_per_class, replace=False)

            vals, bases = [], []
            for n_saved, i in enumerate(rows):
                v, b = shap_explain_instance(model, img_feat[i:i + 1], gnn_feat[i:i + 1], scalar[i],
                                             masker, head, int(target_all[i]), device)
                vals.append(v)
                bases.append(b)
                if n_saved < args.max_images_per_class:
                    save_waterfall(
                        v, b, scalar_raw[i],
                        per_image_dir / f"{head}_{_safe(cls_name)}_{_safe(image_names[i])}.png",
                        title=f"{head}={cls_name}  |  {image_names[i]}  "
                              f"(target={args.target}, p={probs[i, target_all[i]]:.2f})",
                    )
            vals = np.stack(vals)
            bases = np.array(bases)
            all_vals.append(vals); all_base.append(bases); all_rows.append(rows)
            class_rows.append(vals.mean(axis=0)); class_names.append(cls_name)

            save_beeswarm(vals, bases, scalar_raw[rows],
                          aggregate_dir / f"{head}_{_safe(cls_name)}_beeswarm.png",
                          title=f"SHAP — {head}={cls_name}  (n={len(rows)}, target={args.target})")

            mean_abs = np.abs(vals).mean(axis=0)
            head_summary["classes"][str(cls_name)] = {
                "n_examples": int(len(rows)),
                "mean_base_value": float(bases.mean()),
                "shap_mean": {k: float(x) for k, x in zip(SCALAR_COLS, vals.mean(axis=0))},
                "shap_mean_abs": {k: float(x) for k, x in zip(SCALAR_COLS, mean_abs)},
                "ranking": [SCALAR_COLS[k] for k in np.argsort(-mean_abs)],
            }
            print(f"[{head}={cls_name}] n={len(rows)}  top feature: {SCALAR_COLS[int(np.argmax(mean_abs))]}")

        if not all_vals:
            print(f"[WARN] No class in head '{head}' had >= {args.min_examples_per_class} examples; skipped.")
            continue

        vals = np.concatenate(all_vals)
        bases = np.concatenate(all_base)
        rows = np.concatenate(all_rows)
        shap_mean_abs = np.abs(vals).mean(axis=0)
        shap_std_abs = np.abs(vals).std(axis=0)

        # Local accuracy check: base + sum(SHAP) must reproduce the explained probability
        explained_p = probs[rows, target_all[rows]]
        additivity_err = float(np.abs(bases + vals.sum(axis=1) - explained_p).max())

        save_beeswarm(vals, bases, scalar_raw[rows], aggregate_dir / f"{head}_all_beeswarm.png",
                      title=f"SHAP — {head}, all classes  (n={len(rows)}, target={args.target})")
        save_class_heatmap(np.stack(class_rows), class_names, aggregate_dir / f"{head}_class_heatmap.png",
                           title=f"Mean signed SHAP per class — {head}  (target={args.target})")
        save_comparison_figure(shap_mean_abs, shap_std_abs, perm["prob_drop_mean"], perm["prob_drop_std"],
                               aggregate_dir / f"{head}_shap_vs_permutation_prob.png",
                               title=f"SHAP vs permutation importance (Δ target prob) — {head}",
                               perm_label="Permutation (Δ target prob)")
        save_comparison_figure(shap_mean_abs, shap_std_abs, perm["acc_drop_mean"], perm["acc_drop_std"],
                               aggregate_dir / f"{head}_shap_vs_permutation_acc.png",
                               title=f"SHAP vs permutation importance (Δ accuracy) — {head}",
                               perm_label="Permutation (Δ accuracy)")

        rho_p, pval_p, r_p = _corr(shap_mean_abs, perm["prob_drop_mean"])
        rho_a, pval_a, r_a = _corr(shap_mean_abs, perm["acc_drop_mean"])

        head_summary.update({
            "n_explained": int(len(rows)),
            "shap_additivity_max_abs_error": additivity_err,
            "shap_mean_abs": {k: float(x) for k, x in zip(SCALAR_COLS, shap_mean_abs)},
            "shap_ranking": [SCALAR_COLS[k] for k in np.argsort(-shap_mean_abs)],
            "permutation": {
                "n_examples": int(len(scalar)),
                "n_repeats": int(args.n_repeats),
                "base_accuracy": perm["base_accuracy"],
                "base_target_prob": perm["base_target_prob"],
                "acc_drop_mean": {k: float(x) for k, x in zip(SCALAR_COLS, perm["acc_drop_mean"])},
                "acc_drop_std": {k: float(x) for k, x in zip(SCALAR_COLS, perm["acc_drop_std"])},
                "prob_drop_mean": {k: float(x) for k, x in zip(SCALAR_COLS, perm["prob_drop_mean"])},
                "prob_drop_std": {k: float(x) for k, x in zip(SCALAR_COLS, perm["prob_drop_std"])},
                "ranking_prob": [SCALAR_COLS[k] for k in np.argsort(-perm["prob_drop_mean"])],
            },
            "agreement_vs_permutation_prob": {"spearman": rho_p, "spearman_pvalue": pval_p, "pearson": r_p},
            "agreement_vs_permutation_acc": {"spearman": rho_a, "spearman_pvalue": pval_a, "pearson": r_a},
        })
        summary[head] = head_summary
        print(f"[{head}] SHAP ranking: {head_summary['shap_ranking']}")
        print(f"[{head}] SHAP vs permutation (Δprob) Spearman={rho_p}  |  additivity max err={additivity_err:.2e}")

    with open(out_dir / "shap_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {out_dir / 'shap_comparison.json'}")
    print(f"[INFO] Per-image figures : {per_image_dir}")
    print(f"[INFO] Aggregate figures : {aggregate_dir}")


if __name__ == "__main__":
    main()
