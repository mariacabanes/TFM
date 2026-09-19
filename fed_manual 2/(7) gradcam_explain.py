"""
(7) gradcam_explain.py — Grad-CAM vs Grad-CAM++ over the EfficientNetB2 image branch
of the trained ImplantClassifier.

Implements, side by side, from the *same* forward/backward pass and target layer:

    - Grad-CAM     (Selvaraju et al. 2017, "Visual Explanations from Deep Networks
                     via Gradient-based Localization")
    - Grad-CAM++   (Chattopadhay et al. 2018, "Generalized Gradient-based Visual
                     Explanations for Deep Convolutional Networks")

`captum` (the library named for this branch in the README) ships `LayerGradCam` but
has no Grad-CAM++ implementation. To keep the two techniques directly comparable —
same activation map, same gradient, same normalization — both are computed here from
one shared forward-hook capture instead of mixing a captum call for one with a custom
implementation for the other.

Target layer: `model.cnn_backbone.features`, the last spatial feature map of the
EfficientNetB2 backbone (shape (B, 1408, H, W)), right before the global-average-pool
that TCAV (step 10) probes as a flat vector. This is the standard Grad-CAM target: the
last convolutional layer, before spatial information is discarded.

For each brand/diameter class with enough test examples, this script:
    1. Saves a handful of per-image comparison figures (input crop | Grad-CAM overlay
       | Grad-CAM++ overlay).
    2. Saves an aggregate mean saliency map per method (where the model looks, on
       average, when predicting that class).
    3. Reports quantitative agreement between the two methods (cosine similarity,
       Pearson correlation, IoU of the top-20% activated region) — the "comparison"
       the two techniques are meant to provide against each other.

This script only *reads* the checkpoint trained by `fed_manual 1`'s `(6) train.py`;
it never updates the model in place.

Usage:
    python "(7) gradcam_explain.py"
    python "(7) gradcam_explain.py" --heads brand --target true --max-images-per-class 10
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader

from common_explain import (
    SEED, HERE, ActivationGrabber, ImplantProbeDataset,
    build_edge_index, build_label_maps_and_scaler, find_fed1_dir, find_train_csv,
    load_model, load_probe_df,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


# =============================================================================
#  GRAD-CAM / GRAD-CAM++  (shared activation + gradient -> two attribution maps)
# =============================================================================

def compute_cams(A: torch.Tensor, G: torch.Tensor, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    A, G: activations / gradients of the target layer, shape (B, C, H, W).
    Returns (cam_gradcam, cam_gradcam_pp), each (B, *image_size) in [0, 1].
    """
    # ---- Grad-CAM: weights = global-average-pooled gradient ----
    weights_gc = G.mean(dim=(2, 3), keepdim=True)                      # (B, C, 1, 1)
    cam_gc = torch.relu((weights_gc * A).sum(dim=1))                   # (B, H, W)

    # ---- Grad-CAM++ (eq. 19 in Chattopadhay et al. 2018) ----
    g2 = G ** 2
    g3 = g2 * G
    sum_A = A.sum(dim=(2, 3), keepdim=True)                            # (B, C, 1, 1)
    eps = 1e-8
    alpha = g2 / (2 * g2 + sum_A * g3 + eps)
    alpha = torch.where(G != 0, alpha, torch.zeros_like(alpha))
    weights_gcpp = (torch.relu(G) * alpha).sum(dim=(2, 3), keepdim=True)
    cam_gcpp = torch.relu((weights_gcpp * A).sum(dim=1))

    return _normalize_resize(cam_gc, image_size), _normalize_resize(cam_gcpp, image_size)


def _normalize_resize(cam: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    cam = TF.interpolate(cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
    B = cam.shape[0]
    flat = cam.view(B, -1)
    cmin = flat.min(dim=1, keepdim=True)[0]
    cmax = flat.max(dim=1, keepdim=True)[0]
    norm = (flat - cmin) / (cmax - cmin + 1e-8)
    return norm.view(B, *size).detach().cpu().numpy()


# =============================================================================
#  VISUALIZATION
# =============================================================================

def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * IMG_STD + IMG_MEAN, 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay_heatmap(base_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * base_rgb_uint8).astype(np.uint8)


def save_comparison_figure(base_rgb, overlay_gc, overlay_gcpp, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, img, subtitle in zip(axes, [base_rgb, overlay_gc, overlay_gcpp],
                                  ["Input crop", "Grad-CAM", "Grad-CAM++"]):
        ax.imshow(img)
        ax.set_title(subtitle)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_aggregate_figure(mean_gc, mean_gcpp, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, img, subtitle in zip(axes, [mean_gc, mean_gcpp], ["Grad-CAM (mean)", "Grad-CAM++ (mean)"]):
        im = ax.imshow(img, cmap="jet", vmin=0, vmax=1)
        ax.set_title(subtitle)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
#  AGREEMENT METRICS BETWEEN THE TWO METHODS
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def topk_iou(a: np.ndarray, b: np.ndarray, frac: float = 0.2) -> float:
    a_mask = a >= np.quantile(a, 1 - frac)
    b_mask = b >= np.quantile(b, 1 - frac)
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a_mask, b_mask).sum() / union)


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Grad-CAM vs Grad-CAM++ for the ImplantClassifier's image branch.")
    p.add_argument("--split", default="test", help="Split used as class-conditional probe examples.")
    p.add_argument("--heads", default="brand,diameter", help="Comma-separated subset of {brand,diameter}.")
    p.add_argument("--target", choices=["predicted", "true"], default="predicted",
                    help="Explain the model's own prediction, or the ground-truth label.")
    p.add_argument("--model", default=None, help="Path to a specific checkpoint (.pt). Default: latest in fed_manual 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--min-examples-per-class", type=int, default=5)
    p.add_argument("--max-examples-per-class", type=int, default=40,
                    help="Cap on examples used for the aggregate heatmap / comparison metrics.")
    p.add_argument("--max-images-per-class", type=int, default=6,
                    help="Cap on individual per-image comparison figures saved per class.")
    p.add_argument("--overlay-alpha", type=float, default=0.45)
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
    print(f"[INFO] Checkpoint: {ckpt_path}")

    probe_df, probe_split_dir = load_probe_df(fed2_dir, args.split, brand_map, diameter_map, scaler)
    print(f"[INFO] Probe split '{args.split}': {len(probe_df)} usable examples.")

    heads_wanted = set(args.heads.split(","))
    out_dir = fed2_dir / "outputs" / "explanations" / "gradcam"
    per_image_dir = out_dir / "per_image"
    aggregate_dir = out_dir / "aggregate"
    per_image_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    grabber = ActivationGrabber(model.cnn_backbone.features)
    summary = {}

    try:
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
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

                all_gc, all_gcpp = [], []
                saved_images = 0

                for images, node_feats, scalar_feat, brand_idx, diameter_idx, image_names in loader:
                    images = images.to(device)
                    node_feats = node_feats.to(device)
                    scalar_feat = scalar_feat.to(device)
                    B = images.shape[0]
                    batch_vec = torch.arange(B, device=device).repeat_interleave(6)
                    edge_index = build_edge_index(6, B, device)

                    model.zero_grad(set_to_none=True)
                    out_b, out_d = model(images, node_feats.view(-1, 2), edge_index, batch_vec, scalar_feat)
                    logits = out_b if head == "brand" else out_d

                    if args.target == "true":
                        target_idx = (brand_idx if head == "brand" else diameter_idx).to(device)
                    else:
                        target_idx = logits.argmax(dim=1)

                    selected = logits[torch.arange(B, device=device), target_idx].sum()
                    selected.backward()

                    A = grabber.activation.detach()
                    G = grabber.activation.grad.detach()
                    cam_gc, cam_gcpp = compute_cams(A, G, image_size=tuple(images.shape[2:]))
                    all_gc.append(cam_gc)
                    all_gcpp.append(cam_gcpp)

                    for b in range(B):
                        if saved_images >= args.max_images_per_class:
                            continue
                        base = tensor_to_uint8(images[b])
                        ov_gc = overlay_heatmap(base, cam_gc[b], args.overlay_alpha)
                        ov_gcpp = overlay_heatmap(base, cam_gcpp[b], args.overlay_alpha)
                        fname = f"{head}_{_safe(cls_name)}_{_safe(image_names[b])}.png"
                        save_comparison_figure(
                            base, ov_gc, ov_gcpp, per_image_dir / fname,
                            title=f"{head}={cls_name}  |  {image_names[b]}  (target={args.target})",
                        )
                        saved_images += 1

                all_gc = np.concatenate(all_gc, axis=0)
                all_gcpp = np.concatenate(all_gcpp, axis=0)

                cos_sims = [cosine_similarity(all_gc[i], all_gcpp[i]) for i in range(len(all_gc))]
                pearsons = [pearson_corr(all_gc[i], all_gcpp[i]) for i in range(len(all_gc))]
                ious = [topk_iou(all_gc[i], all_gcpp[i]) for i in range(len(all_gc))]

                save_aggregate_figure(
                    all_gc.mean(axis=0), all_gcpp.mean(axis=0),
                    aggregate_dir / f"{head}_{_safe(cls_name)}.png",
                    title=f"Aggregate saliency — {head}={cls_name}  (n={len(all_gc)}, target={args.target})",
                )

                summary[head][str(cls_name)] = {
                    "n_examples": int(len(all_gc)),
                    "cosine_similarity_mean": float(np.mean(cos_sims)),
                    "cosine_similarity_std": float(np.std(cos_sims)),
                    "pearson_corr_mean": float(np.mean(pearsons)),
                    "pearson_corr_std": float(np.std(pearsons)),
                    "topk20_iou_mean": float(np.mean(ious)),
                    "topk20_iou_std": float(np.std(ious)),
                }
                print(f"[{head}={cls_name}] n={len(all_gc)}  "
                      f"cosine={np.mean(cos_sims):.2f}±{np.std(cos_sims):.2f}  "
                      f"pearson={np.mean(pearsons):.2f}  IoU@20%={np.mean(ious):.2f}")
    finally:
        grabber.remove()

    with open(out_dir / "gradcam_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {out_dir / 'gradcam_comparison.json'}")
    print(f"[INFO] Per-image figures : {per_image_dir}")
    print(f"[INFO] Aggregate figures : {aggregate_dir}")


if __name__ == "__main__":
    main()
