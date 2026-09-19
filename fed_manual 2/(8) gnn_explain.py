"""
(8) gnn_explain.py — GNNExplainer vs. counterfactual perturbation of node coordinates,
over the graph branch (6 anatomical keypoints) of the trained ImplantClassifier.

Two independent ways of asking "which keypoint matters most for this prediction?":

    - GNNExplainer (Ying et al. 2019, "GNNExplainer: Generating Explanations for
      Graph Neural Networks", NeurIPS). Learns a soft mask over the graph's edges
      (and a global mask over the 2 coordinate dims) that keeps the target class's
      predicted log-probability high while staying sparse/discrete, with the trained
      model itself frozen. Node importance is read off as the mean importance of the
      edges incident to that node.

    - Counterfactual perturbation of node coordinates. For each keypoint, replace its
      (x, y) with the centroid of the other 5 keypoints ("what if this point sat where
      the rest of the implant says it should") and measure how much the target class's
      predicted probability drops. A model-agnostic, optimization-free alternative.

`torch_geometric.explain.GNNExplainer` is built for a plain `model(x, edge_index)`
classifier; this model's graph branch is one of three branches feeding a shared head
(image + scalar features are the other two), so wrapping it would need as much glue
code as implementing the published algorithm directly against the two GCNConv layers
(which DO accept `edge_weight`, even though `ImplantClassifier.forward` doesn't expose
it) — done here, following the same "read-only reuse of the trained submodules"
approach as steps 7 and 10.

For efficiency, the CNN and scalar branches are only run once per example (frozen,
img_feat/scalar_emb are constants for the graph-mask optimization) — mathematically
identical to a full forward every step, just without redoing the EfficientNetB2 pass
150 times per example.

This script only *reads* the checkpoint trained by `fed_manual 1`'s `(6) train.py`;
it never updates the model in place.

Usage:
    python "(8) gnn_explain.py"
    python "(8) gnn_explain.py" --heads brand --target true --max-examples-per-class 20
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as TF
from torch_geometric.nn import global_mean_pool
from scipy.stats import spearmanr, pearsonr

from explain_common import (
    SEED, HERE, NODE_KEYS, ImplantProbeDataset,
    build_edge_index, build_label_maps_and_scaler, find_fed1_dir, find_train_csv,
    load_model, load_probe_df,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_NODES = len(NODE_KEYS)          # 6 anatomical keypoints
# torch_geometric's dense_to_sparse (used by build_edge_index) visits (i, j) pairs in this
# exact row-major order, so this list indexes the flat edge_importance arrays everywhere below.
EDGE_INDEX_PAIRS = [(i, j) for i in range(N_NODES) for j in range(N_NODES) if i != j]

# GNNExplainer regularization coefficients (same defaults as PyG's reference implementation)
EDGE_SIZE_COEF = 0.005
EDGE_ENT_COEF = 1.0
FEAT_SIZE_COEF = 1.0
FEAT_ENT_COEF = 0.1
EPS = 1e-8


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


# =============================================================================
#  SHARED GNN-BRANCH FORWARD  (mirrors ImplantClassifier.forward, edge_weight-aware)
# =============================================================================

def gnn_branch_forward(model, node_feats, edge_index, batch_vec, edge_weight, img_feat, scalar_emb):
    """Same computation as ImplantClassifier.forward from the GNN branch onward,
    with img_feat/scalar_emb precomputed (frozen, don't depend on the graph mask)."""
    x = torch.relu(model.gcn1(node_feats, edge_index, edge_weight=edge_weight))
    x = model.gcn2(x, edge_index, edge_weight=edge_weight)
    gnn_feat = global_mean_pool(x, batch_vec)
    combined = torch.cat([img_feat, gnn_feat, scalar_emb], dim=1)
    out = model.fc(combined)
    return model.brand_head(out), model.diameter_head(out)


def _mask_entropy(m: torch.Tensor) -> torch.Tensor:
    m = m.clamp(EPS, 1 - EPS)
    return (-(m * torch.log(m) + (1 - m) * torch.log(1 - m))).mean()


# =============================================================================
#  METHOD 1 — GNNExplainer  (Ying et al. 2019)
# =============================================================================

def gnn_explain_instance(model, image: torch.Tensor, node_feats: torch.Tensor, scalar_feat: torch.Tensor,
                          head: str, class_idx: int, device, n_steps: int, lr: float):
    """Returns (node_importance[6], edge_importance[30], feat_importance[2])."""
    edge_index = build_edge_index(N_NODES, 1, device)   # (2, 30) for one graph
    n_edges = edge_index.shape[1]
    batch_vec = torch.zeros(N_NODES, dtype=torch.long, device=device)

    image_b = image.unsqueeze(0).to(device)
    scalar_b = scalar_feat.unsqueeze(0).to(device)
    node_feats_dev = node_feats.to(device)

    with torch.no_grad():
        img_feat = model.cnn_fc(model.cnn_backbone(image_b))
        scalar_emb = model.scalar_fc(scalar_b)

    edge_mask_logit = torch.nn.Parameter(torch.randn(n_edges, device=device) * 0.1)
    feat_mask_logit = torch.nn.Parameter(torch.randn(node_feats_dev.shape[1], device=device) * 0.1)
    optimizer = torch.optim.Adam([edge_mask_logit, feat_mask_logit], lr=lr)
    target = torch.tensor([class_idx], device=device)

    for _ in range(n_steps):
        optimizer.zero_grad()
        edge_weight = torch.sigmoid(edge_mask_logit)
        feat_weight = torch.sigmoid(feat_mask_logit)
        masked_feats = node_feats_dev * feat_weight.unsqueeze(0)

        out_b, out_d = gnn_branch_forward(model, masked_feats, edge_index, batch_vec,
                                           edge_weight, img_feat, scalar_emb)
        logits = out_b if head == "brand" else out_d

        loss = (
            TF.cross_entropy(logits, target)
            + EDGE_SIZE_COEF * edge_weight.sum()
            + EDGE_ENT_COEF * _mask_entropy(edge_weight)
            + FEAT_SIZE_COEF * feat_weight.mean()
            + FEAT_ENT_COEF * _mask_entropy(feat_weight)
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        edge_importance = torch.sigmoid(edge_mask_logit).cpu().numpy()
        feat_importance = torch.sigmoid(feat_mask_logit).cpu().numpy()

    src, dst = edge_index.cpu().numpy()
    node_importance = np.zeros(N_NODES)
    for k in range(N_NODES):
        incident = (src == k) | (dst == k)
        node_importance[k] = edge_importance[incident].mean() if incident.any() else 0.0

    return node_importance, edge_importance, feat_importance


# =============================================================================
#  METHOD 2 — Counterfactual perturbation of node coordinates
# =============================================================================

@torch.no_grad()
def counterfactual_node_importance(model, image: torch.Tensor, node_feats: torch.Tensor,
                                    scalar_feat: torch.Tensor, head: str, class_idx: int, device):
    """For each node, moves it to the centroid of the other 5 and measures the drop
    in predicted probability for the target class. Larger drop = more important node."""
    edge_index = build_edge_index(N_NODES, 1, device)
    batch_vec = torch.zeros(N_NODES, dtype=torch.long, device=device)
    image_b = image.unsqueeze(0).to(device)
    scalar_b = scalar_feat.unsqueeze(0).to(device)
    node_feats_dev = node_feats.to(device)

    def predict_prob(feats):
        out_b, out_d = model(image_b, feats, edge_index, batch_vec, scalar_b)
        logits = out_b if head == "brand" else out_d
        return TF.softmax(logits, dim=1)[0, class_idx].item()

    base_prob = predict_prob(node_feats_dev)

    importance = np.zeros(N_NODES)
    for k in range(N_NODES):
        others = torch.cat([node_feats_dev[:k], node_feats_dev[k + 1:]], dim=0)
        centroid = others.mean(dim=0)
        perturbed = node_feats_dev.clone()
        perturbed[k] = centroid
        importance[k] = base_prob - predict_prob(perturbed)

    return importance, base_prob


# =============================================================================
#  VISUALIZATION
# =============================================================================

def save_instance_figure(node_feats: np.ndarray, edge_importance: np.ndarray,
                          gnn_node_imp: np.ndarray, cf_node_imp: np.ndarray,
                          out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- Left: keypoint skeleton, edges/nodes weighted by GNNExplainer importance ----
    ax = axes[0]
    edge_imp_norm = (edge_importance - edge_importance.min()) / (np.ptp(edge_importance) + 1e-8)
    idx = 0
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i == j:
                continue
            w = edge_imp_norm[idx]
            idx += 1
            ax.plot([node_feats[i, 0], node_feats[j, 0]], [node_feats[i, 1], node_feats[j, 1]],
                    color="crimson", alpha=0.15 + 0.6 * w, linewidth=0.5 + 2.5 * w, zorder=1)
    node_size = 80 + 400 * (gnn_node_imp - gnn_node_imp.min()) / (np.ptp(gnn_node_imp) + 1e-8)
    ax.scatter(node_feats[:, 0], node_feats[:, 1], s=node_size, c="steelblue", zorder=2, edgecolors="black")
    for k, name in enumerate(NODE_KEYS):
        ax.annotate(name, (node_feats[k, 0], node_feats[k, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.invert_yaxis()
    ax.set_title("GNNExplainer\n(node size / edge width = importance)", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    # ---- Right: grouped bar chart comparing both methods per keypoint ----
    ax = axes[1]
    x = np.arange(N_NODES)
    width = 0.38
    ax.bar(x - width / 2, gnn_node_imp, width, label="GNNExplainer", color="#2a9d8f")
    ax.bar(x + width / 2, cf_node_imp, width, label="Counterfactual (Δprob)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(NODE_KEYS, rotation=30, ha="right", fontsize=7)
    ax.set_title("Per-keypoint importance — both methods", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_aggregate_figure(gnn_mean, gnn_std, cf_mean, cf_std, edge_matrix_mean,
                           out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    x = np.arange(N_NODES)
    width = 0.38
    ax.bar(x - width / 2, gnn_mean, width, yerr=gnn_std, label="GNNExplainer", color="#2a9d8f", capsize=3)
    ax.bar(x + width / 2, cf_mean, width, yerr=cf_std, label="Counterfactual (Δprob)", color="#e76f51", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(NODE_KEYS, rotation=30, ha="right", fontsize=7)
    ax.set_title("Mean keypoint importance", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    im = ax.imshow(edge_matrix_mean, cmap="viridis", vmin=0, vmax=edge_matrix_mean.max() + 1e-8)
    ax.set_xticks(range(N_NODES)); ax.set_xticklabels(NODE_KEYS, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(N_NODES)); ax.set_yticklabels(NODE_KEYS, fontsize=6)
    ax.set_title("GNNExplainer mean edge importance", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GNNExplainer vs counterfactual node-coordinate perturbation.")
    p.add_argument("--split", default="test", help="Split used as class-conditional probe examples.")
    p.add_argument("--heads", default="brand,diameter", help="Comma-separated subset of {brand,diameter}.")
    p.add_argument("--target", choices=["predicted", "true"], default="predicted",
                    help="Explain the model's own prediction, or the ground-truth label.")
    p.add_argument("--model", default=None, help="Path to a specific checkpoint (.pt). Default: latest in fed_manual 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--min-examples-per-class", type=int, default=5)
    p.add_argument("--max-examples-per-class", type=int, default=15,
                    help="Cap on examples per class (GNNExplainer optimizes a mask per example — keep this modest).")
    p.add_argument("--max-images-per-class", type=int, default=4,
                    help="Cap on individual per-example comparison figures saved per class.")
    p.add_argument("--n-steps", type=int, default=150, help="Adam steps per GNNExplainer instance optimization.")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate for the GNNExplainer mask optimizer.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    fed1_dir = find_fed1_dir()
    fed2_dir = HERE
    print(f"[INFO] fed_manual 1 (models, read-only): {fed1_dir}")
    print(f"[INFO] fed_manual 2 (probes)            : {fed2_dir}")

    brand_map, diameter_map, scaler = build_label_maps_and_scaler(find_train_csv(fed1_dir))
    model, ckpt_path = load_model(fed1_dir, brand_map, diameter_map, device, args.model)
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[INFO] Checkpoint: {ckpt_path}")

    probe_df, probe_split_dir = load_probe_df(fed2_dir, args.split, brand_map, diameter_map, scaler)
    print(f"[INFO] Probe split '{args.split}': {len(probe_df)} usable examples.")

    heads_wanted = set(args.heads.split(","))
    out_dir = fed2_dir / "outputs" / "explanations" / "gnn_explainer"
    per_image_dir = out_dir / "per_image"
    aggregate_dir = out_dir / "aggregate"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for head, label_map, idx_col in (
        ("brand", brand_map, "brand_idx"),
        ("diameter", diameter_map, "diameter_idx"),
    ):
        if head not in heads_wanted:
            continue
        summary[head] = {}

        for cls_name, cls_idx in label_map.items():
            subset = probe_df[probe_df[idx_col] == cls_idx]
            if len(subset) < args.min_examples_per_class:
                continue
            subset = subset.sample(min(len(subset), args.max_examples_per_class), random_state=SEED)

            ds = ImplantProbeDataset(subset, probe_split_dir / "cropped_images", probe_split_dir / "masks")

            gnn_node_imps, cf_node_imps, edge_matrices = [], [], []
            per_example_spearman = []
            saved_images = 0

            for i in range(len(ds)):
                image, node_feats, scalar_feat, brand_idx, diameter_idx, image_name = ds[i]
                target_idx = (brand_idx if head == "brand" else diameter_idx) if args.target == "true" else None

                if target_idx is None:
                    with torch.no_grad():
                        edge_index = build_edge_index(N_NODES, 1, device)
                        batch_vec = torch.zeros(N_NODES, dtype=torch.long, device=device)
                        out_b, out_d = model(image.unsqueeze(0).to(device), node_feats.to(device),
                                              edge_index, batch_vec, scalar_feat.unsqueeze(0).to(device))
                        logits = out_b if head == "brand" else out_d
                        target_idx = int(logits.argmax(dim=1).item())

                gnn_node_imp, edge_imp, _feat_imp = gnn_explain_instance(
                    model, image, node_feats, scalar_feat, head, target_idx, device, args.n_steps, args.lr,
                )
                cf_node_imp, base_prob = counterfactual_node_importance(
                    model, image, node_feats, scalar_feat, head, target_idx, device,
                )

                gnn_node_imps.append(gnn_node_imp)
                cf_node_imps.append(cf_node_imp)
                edge_matrices.append(edge_imp)

                if np.std(gnn_node_imp) > 1e-8 and np.std(cf_node_imp) > 1e-8:
                    rho, _ = spearmanr(gnn_node_imp, cf_node_imp)
                    if not np.isnan(rho):
                        per_example_spearman.append(rho)

                if saved_images < args.max_images_per_class:
                    save_instance_figure(
                        node_feats.numpy(), edge_imp, gnn_node_imp, cf_node_imp,
                        per_image_dir / f"{head}_{_safe(cls_name)}_{_safe(image_name)}.png",
                        title=f"{head}={cls_name}  |  {image_name}  (target={args.target}, base_prob={base_prob:.2f})",
                    )
                    saved_images += 1

            gnn_node_imps = np.stack(gnn_node_imps)   # (N, 6)
            cf_node_imps = np.stack(cf_node_imps)     # (N, 6)
            edge_matrix_mean = np.zeros((N_NODES, N_NODES))
            mean_edge_imp = np.stack(edge_matrices).mean(axis=0)
            for (i, j), val in zip(EDGE_INDEX_PAIRS, mean_edge_imp):
                edge_matrix_mean[i, j] = val

            gnn_mean, gnn_std = gnn_node_imps.mean(axis=0), gnn_node_imps.std(axis=0)
            cf_mean, cf_std = cf_node_imps.mean(axis=0), cf_node_imps.std(axis=0)

            rho_agg, pval_agg = spearmanr(gnn_mean, cf_mean)
            r_agg, _ = pearsonr(gnn_mean, cf_mean)

            save_aggregate_figure(
                gnn_mean, gnn_std, cf_mean, cf_std, edge_matrix_mean,
                aggregate_dir / f"{head}_{_safe(cls_name)}.png",
                title=f"Aggregate keypoint importance — {head}={cls_name}  (n={len(gnn_node_imps)}, target={args.target})",
            )

            summary[head][str(cls_name)] = {
                "n_examples": int(len(gnn_node_imps)),
                "gnn_explainer_mean": {k: float(v) for k, v in zip(NODE_KEYS, gnn_mean)},
                "counterfactual_mean": {k: float(v) for k, v in zip(NODE_KEYS, cf_mean)},
                "aggregate_spearman": float(rho_agg) if not np.isnan(rho_agg) else None,
                "aggregate_spearman_pvalue": float(pval_agg) if not np.isnan(pval_agg) else None,
                "aggregate_pearson": float(r_agg) if not np.isnan(r_agg) else None,
                "mean_per_example_spearman": float(np.mean(per_example_spearman)) if per_example_spearman else None,
            }
            print(f"[{head}={cls_name}] n={len(gnn_node_imps)}  "
                  f"aggregate Spearman={rho_agg:.2f} (p={pval_agg:.3f})  "
                  f"mean per-example Spearman={np.mean(per_example_spearman) if per_example_spearman else float('nan'):.2f}")

    with open(out_dir / "gnn_explain_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {out_dir / 'gnn_explain_comparison.json'}")
    print(f"[INFO] Per-image figures : {per_image_dir}")
    print(f"[INFO] Aggregate figures : {aggregate_dir}")


if __name__ == "__main__":
    main()
