"""
(7b) ig_explain.py — Integrated Gradients over the EfficientNetB2 image branch of the
trained ImplantClassifier, compared against Grad-CAM (step 7).

    - Integrated Gradients (Sundararajan et al. 2017, "Axiomatic Attribution for Deep
      Networks", ICML). For an input x, a baseline x0 and a target class c:

          IG_i(x) = (x_i − x0_i) · ∫₀¹ ∂f_c(x0 + α(x − x0)) / ∂x_i  dα

      approximated with a midpoint Riemann sum over `steps` points on the straight
      path from x0 to x. f_c is the target-class logit. The per-pixel map is the sum
      over the 3 RGB channels, and it is SIGNED: positive pixels push the model towards
      class c, negative pixels push it away.

Why it complements Grad-CAM (step 7):
    - Resolution: Grad-CAM lives on EfficientNet's last 9×9 feature map and is
      upsampled; IG attributes to the 260×260 input pixels directly, so it can point
      at fine structures (thread edges, implant outline, apex).
    - Sign: Grad-CAM keeps only positive evidence (ReLU); IG keeps both.
    - Axioms: IG satisfies completeness, Σ_i IG_i(x) = f_c(x) − f_c(x0). This script
      checks it on every example (relative completeness error), the same way step 9
      checks SHAP additivity.

Baseline: by default an all-black image (0 in pixel space, then ImageNet-normalized).
The crops are mask-multiplied at train time, so black is exactly the value of
"no implant" pixels — background pixels get zero attribution by construction.
`--baseline blur` / `gray` are available to test sensitivity to that choice.

Only the image branch is differentiated: the graph and scalar embeddings are computed
once per example and held fixed along the path — mathematically identical to a full
forward per step (the same approach as steps 8 and 9).

`captum` ships `IntegratedGradients`, but it expects a single-input model. This model
has five inputs and only the image should vary, so IG is implemented directly (a few
lines) to keep the same "read-only reuse of the trained submodules" approach as steps
7–10. `(12) evaluate_explanations.py` imports `integrated_gradients` from this file,
so the evaluated explanation is exactly the one produced here.

For each brand/diameter class with enough test examples, this script:
    1. Saves per-image figures: input crop | |IG| overlay | signed IG | Grad-CAM overlay.
    2. Saves an aggregate mean |IG| map per class (next to the mean Grad-CAM map).
    3. Reports agreement between IG and Grad-CAM (Spearman, Pearson, top-20% IoU,
       computed on Gaussian-smoothed |IG| so both maps have comparable spatial scale),
       the share of positive vs negative attribution, and the completeness error.

This script only *reads* the checkpoint trained by `fed_manual 1`'s `(6) train.py`;
it never updates the model in place.

Usage:
    python "(7b) ig_explain.py"
    python "(7b) ig_explain.py" --heads brand --target true --steps 64 --baseline blur
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as TVF
from torch_geometric.nn import global_mean_pool
from scipy.stats import spearmanr

from common_explain import (
    SEED, HERE, NODE_KEYS, ActivationGrabber, ImplantProbeDataset,
    build_edge_index, build_label_maps_and_scaler, find_fed1_dir, find_train_csv,
    load_model, load_probe_df,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_NODES = len(NODE_KEYS)
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BASELINES = ("black", "blur", "gray")


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _load_step_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
#  MODEL PIECES  (image branch with graph/scalar embeddings held fixed)
# =============================================================================

@torch.no_grad()
def side_embeddings(model, node_feats, scalar_feat, device):
    """Graph + scalar embeddings of one example (constants for the image attribution)."""
    edge_index = build_edge_index(N_NODES, 1, device)
    batch_vec = torch.zeros(N_NODES, dtype=torch.long, device=device)
    x = torch.relu(model.gcn1(node_feats.to(device), edge_index))
    x = model.gcn2(x, edge_index)
    gnn_feat = global_mean_pool(x, batch_vec)
    scalar_emb = model.scalar_fc(scalar_feat.unsqueeze(0).to(device))
    return gnn_feat, scalar_emb


def image_branch_logits(model, images, gnn_feat, scalar_emb, head):
    """Same as ImplantClassifier.forward with the graph/scalar embeddings precomputed."""
    B = images.shape[0]
    img_feat = model.cnn_fc(model.cnn_backbone(images))
    combined = torch.cat([img_feat, gnn_feat.expand(B, -1), scalar_emb.expand(B, -1)], dim=1)
    out = model.fc(combined)
    return model.brand_head(out) if head == "brand" else model.diameter_head(out)


# =============================================================================
#  INTEGRATED GRADIENTS
# =============================================================================

def make_baseline(image: torch.Tensor, kind: str = "black") -> torch.Tensor:
    """Baseline x0 in the model's normalized input space."""
    mean, std = IMG_MEAN.to(image.device), IMG_STD.to(image.device)
    pix = (image * std + mean).clamp(0, 1)
    if kind == "black":
        base = torch.zeros_like(pix)
    elif kind == "gray":
        base = torch.full_like(pix, 0.5)
    elif kind == "blur":
        base = TVF.gaussian_blur(pix.unsqueeze(0), kernel_size=51, sigma=15.0)[0]
    else:
        raise ValueError(f"Unknown baseline '{kind}', expected one of {BASELINES}")
    return (base - mean) / std


def integrated_gradients(model, image, gnn_feat, scalar_emb, head, cls, device,
                         steps: int = 50, batch_size: int = 16, baseline: str = "black",
                         return_details: bool = False):
    """
    IG of the target-class logit w.r.t. the (normalized) input image.

    image: (3, H, W) normalized tensor; gnn_feat / scalar_emb from `side_embeddings`.
    Returns the channel-summed signed attribution map (H, W) as a numpy array.
    With return_details=True also returns a dict with the per-channel attribution,
    f(x), f(x0) and the completeness error |Σ IG − (f(x) − f(x0))| / |f(x) − f(x0)|.
    """
    x = image.to(device)
    x0 = make_baseline(x, baseline)
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps   # midpoint rule
    total_grad = torch.zeros_like(x)
    for s in range(0, steps, batch_size):
        a = alphas[s:s + batch_size].view(-1, 1, 1, 1)
        xs = (x0 + a * (x - x0)).requires_grad_(True)
        logits = image_branch_logits(model, xs, gnn_feat, scalar_emb, head)
        grad, = torch.autograd.grad(logits[:, cls].sum(), xs)
        total_grad += grad.sum(0)
    attr = ((x - x0) * total_grad / steps).detach()
    attr_map = attr.sum(0).cpu().numpy()
    if not return_details:
        return attr_map

    with torch.no_grad():
        f = image_branch_logits(model, torch.stack([x, x0]), gnn_feat, scalar_emb, head)[:, cls]
    f_x, f_x0 = float(f[0]), float(f[1])
    gap = f_x - f_x0
    err = abs(float(attr.sum()) - gap) / (abs(gap) + 1e-8)
    return attr_map, {"attr": attr.cpu().numpy(), "f_x": f_x, "f_x0": f_x0, "completeness_rel_error": err}


# =============================================================================
#  GRAD-CAM (for comparison; same computation as step 7)
# =============================================================================

def gradcam_map(model, GC, image, gnn_feat, scalar_emb, head, cls, device) -> np.ndarray:
    grabber = ActivationGrabber(model.cnn_backbone.features)
    try:
        x = image.unsqueeze(0).to(device).requires_grad_(True)   # model params are frozen
        logits = image_branch_logits(model, x, gnn_feat, scalar_emb, head)
        logits[0, cls].backward()
        A, G = grabber.activation.detach(), grabber.activation.grad.detach()
    finally:
        grabber.remove()
    cam_gc, _ = GC.compute_cams(A, G, image_size=tuple(image.shape[1:]))
    return cam_gc[0]


# =============================================================================
#  METRICS / VISUALIZATION
# =============================================================================

def abs_normalized(attr_map: np.ndarray, smooth_sigma: float = 0.0, clip_pct: float = 99.0) -> np.ndarray:
    """|IG| scaled to [0, 1] (clipped at a high percentile so a few pixels don't dominate)."""
    m = np.abs(attr_map).astype(np.float32)
    if smooth_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), smooth_sigma)
    hi = np.percentile(m, clip_pct)
    return np.clip(m / (hi + 1e-12), 0, 1)


def spearman(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def pearson(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def topk_iou(a, b, frac: float = 0.2) -> float:
    a_mask = a >= np.quantile(a, 1 - frac)
    b_mask = b >= np.quantile(b, 1 - frac)
    union = np.logical_or(a_mask, b_mask).sum()
    return float(np.logical_and(a_mask, b_mask).sum() / union) if union else 0.0


def tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    arr = (img.detach().cpu() * IMG_STD + IMG_MEAN).clamp(0, 1).numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)


def overlay(base_rgb: np.ndarray, m01: np.ndarray, alpha: float, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    heat = cv2.cvtColor(cv2.applyColorMap((m01 * 255).astype(np.uint8), cmap), cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * base_rgb).astype(np.uint8)


def save_instance_figure(base, ig_abs, ig_signed, cam, out_path: Path, title: str, alpha: float):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(base); axes[0].set_title("Input crop")
    axes[1].imshow(overlay(base, ig_abs, alpha, cv2.COLORMAP_HOT)); axes[1].set_title("|Integrated Gradients|")
    vmax = np.percentile(np.abs(ig_signed), 99) + 1e-12
    im = axes[2].imshow(ig_signed, cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[2].set_title("IG signed (red = for, blue = against)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(overlay(base, cam, alpha)); axes[3].set_title("Grad-CAM (step 7)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_aggregate_figure(mean_ig, mean_signed, mean_cam, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    im = axes[0].imshow(mean_ig, cmap="hot", vmin=0, vmax=1); axes[0].set_title("|IG| (mean)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    v = np.abs(mean_signed).max() + 1e-12
    im = axes[1].imshow(mean_signed, cmap="bwr", vmin=-v, vmax=v); axes[1].set_title("IG signed (mean, per-image scaled)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    im = axes[2].imshow(mean_cam, cmap="jet", vmin=0, vmax=1); axes[2].set_title("Grad-CAM (mean)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Integrated Gradients (vs Grad-CAM) for the ImplantClassifier's image branch.")
    p.add_argument("--split", default="test")
    p.add_argument("--heads", default="brand,diameter", help="Comma-separated subset of {brand,diameter}.")
    p.add_argument("--target", choices=["predicted", "true"], default="predicted")
    p.add_argument("--model", default=None, help="Path to a specific checkpoint (.pt). Default: latest in fed_manual 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int, default=50, help="Riemann steps on the baseline → input path.")
    p.add_argument("--baseline", choices=BASELINES, default="black")
    p.add_argument("--batch-size", type=int, default=16, help="Path points per forward/backward pass.")
    p.add_argument("--min-examples-per-class", type=int, default=5)
    p.add_argument("--max-examples-per-class", type=int, default=40)
    p.add_argument("--max-images-per-class", type=int, default=6)
    p.add_argument("--smooth-sigma", type=float, default=4.0,
                   help="Gaussian sigma (px) applied to |IG| before comparing with Grad-CAM.")
    p.add_argument("--overlay-alpha", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    GC = _load_step_module("gradcam_explain", "(7) gradcam_explain.py")
    fed1_dir = find_fed1_dir()
    brand_map, diameter_map, scaler = build_label_maps_and_scaler(find_train_csv(fed1_dir))
    model, ckpt_path = load_model(fed1_dir, brand_map, diameter_map, device, args.model)
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"[INFO] Checkpoint: {ckpt_path}")

    probe_df, split_dir = load_probe_df(HERE, args.split, brand_map, diameter_map, scaler)
    print(f"[INFO] Probe split '{args.split}': {len(probe_df)} usable examples.")

    out_dir = HERE / "outputs" / "explanations" / "integrated_gradients"
    per_image_dir, aggregate_dir = out_dir / "per_image", out_dir / "aggregate"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    heads_wanted = set(args.heads.split(","))
    summary = {"config": {"steps": args.steps, "baseline": args.baseline, "target": args.target,
                          "smooth_sigma": args.smooth_sigma, "checkpoint": str(ckpt_path)}}

    for head, label_map, idx_col in (("brand", brand_map, "brand_idx"), ("diameter", diameter_map, "diameter_idx")):
        if head not in heads_wanted:
            continue
        summary[head] = {}
        for cls_name, cls_idx in label_map.items():
            subset = probe_df[probe_df[idx_col] == cls_idx]
            if len(subset) < args.min_examples_per_class:
                continue
            subset = subset.sample(min(len(subset), args.max_examples_per_class), random_state=SEED)
            ds = ImplantProbeDataset(subset, split_dir / "cropped_images", split_dir / "masks")

            abs_maps, signed_maps, cams = [], [], []
            sp, pr, ious, pos_share, compl = [], [], [], [], []
            for i in range(len(ds)):
                image, node_feats, scalar_feat, b_idx, d_idx, image_name = ds[i]
                gnn_feat, scalar_emb = side_embeddings(model, node_feats, scalar_feat, device)
                if args.target == "true":
                    target = b_idx if head == "brand" else d_idx
                else:
                    with torch.no_grad():
                        target = int(image_branch_logits(model, image.unsqueeze(0).to(device),
                                                         gnn_feat, scalar_emb, head).argmax(1))

                ig_map, det = integrated_gradients(model, image, gnn_feat, scalar_emb, head, target, device,
                                                   args.steps, args.batch_size, args.baseline, return_details=True)
                cam = gradcam_map(model, GC, image, gnn_feat, scalar_emb, head, target, device)

                ig_abs_smooth = abs_normalized(ig_map, args.smooth_sigma)
                sp.append(spearman(ig_abs_smooth, cam))
                pr.append(pearson(ig_abs_smooth, cam))
                ious.append(topk_iou(ig_abs_smooth, cam))
                total_abs = np.abs(ig_map).sum()
                pos_share.append(float(ig_map[ig_map > 0].sum() / total_abs) if total_abs > 0 else float("nan"))
                compl.append(det["completeness_rel_error"])

                abs_maps.append(abs_normalized(ig_map))
                signed_maps.append(ig_map / (np.abs(ig_map).max() + 1e-12))
                cams.append(cam)

                if i < args.max_images_per_class:
                    save_instance_figure(
                        tensor_to_uint8(image), abs_normalized(ig_map), ig_map, cam,
                        per_image_dir / f"{head}_{_safe(cls_name)}_{_safe(image_name)}.png",
                        title=f"{head}={cls_name}  |  {image_name}  (target={args.target}, baseline={args.baseline}, "
                              f"completeness err={det['completeness_rel_error']:.3f})",
                        alpha=args.overlay_alpha,
                    )

            save_aggregate_figure(
                np.mean(abs_maps, 0), np.mean(signed_maps, 0), np.mean(cams, 0),
                aggregate_dir / f"{head}_{_safe(cls_name)}.png",
                title=f"Aggregate attribution — {head}={cls_name}  (n={len(abs_maps)}, target={args.target})",
            )
            stat = lambda v: (float(np.nanmean(v)), float(np.nanstd(v)))
            summary[head][str(cls_name)] = {
                "n_examples": len(abs_maps),
                "spearman_ig_vs_gradcam_mean": stat(sp)[0], "spearman_ig_vs_gradcam_std": stat(sp)[1],
                "pearson_ig_vs_gradcam_mean": stat(pr)[0], "pearson_ig_vs_gradcam_std": stat(pr)[1],
                "topk20_iou_ig_vs_gradcam_mean": stat(ious)[0], "topk20_iou_ig_vs_gradcam_std": stat(ious)[1],
                "positive_attribution_share_mean": stat(pos_share)[0],
                "positive_attribution_share_std": stat(pos_share)[1],
                "completeness_rel_error_mean": stat(compl)[0], "completeness_rel_error_max": float(np.nanmax(compl)),
            }
            print(f"[{head}={cls_name}] n={len(abs_maps)}  IG vs Grad-CAM: Spearman={stat(sp)[0]:.2f}±{stat(sp)[1]:.2f}  "
                  f"IoU@20%={stat(ious)[0]:.2f}  positive share={stat(pos_share)[0]:.2f}  "
                  f"completeness err={stat(compl)[0]:.3f}")

    with open(out_dir / "ig_explain.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {out_dir / 'ig_explain.json'}")
    print(f"[INFO] Per-image figures : {per_image_dir}")
    print(f"[INFO] Aggregate figures : {aggregate_dir}")


if __name__ == "__main__":
    main()
