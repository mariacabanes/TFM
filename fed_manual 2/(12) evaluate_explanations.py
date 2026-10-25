"""
(12) evaluate_explanations.py — Quantitative reliability evaluation of the explainability
layer (image branch: Grad-CAM / Grad-CAM++ / Integrated Gradients; graph branch:
GNNExplainer / counterfactual node perturbation), run on the test split.

Two families of evaluation, each reported as mean ± SD per brand / diameter class
(and over all classes):

(i) FIDELITY AND SANITY CHECKS
    Image branch — insertion / deletion (Petsiuk et al. 2018, "RISE"):
        - Pixels are ranked by their attribution (highest first, random tie-breaking).
        - Deletion: the ranked pixels are progressively replaced by a neutral value
          (blur, gray or noise) and the target-class probability is tracked.
          Low AUC = good fidelity (the probability collapses fast).
        - Insertion: start from the neutral image and progressively reveal the ranked
          pixels. High AUC = good fidelity.
        - Both curves are always compared against a random pixel ordering baseline
          (same image, same neutral value). The "gain" metrics are
          deletion: AUC_random − AUC_method (>0 is better than random)
          insertion: AUC_method − AUC_random (>0 is better than random).

    Graph branch — same idea over the 6-keypoint graph:
        - Edge deletion / insertion curves (30 directed edges, ranked by the
          GNNExplainer edge mask; removed edges get edge_weight = 0 in both GCNConv
          layers) and node deletion / insertion curves (6 keypoints, ranked by
          GNNExplainer node importance or by the counterfactual Δprob; a removed node
          loses all its incident edges and is excluded from the mean pool).
        - Fidelity+ / Fidelity− (Yuan et al. 2022, "Explainability in Graph Neural
          Networks: A Taxonomic Survey") at several sparsity levels:
              Fidelity+ = p(G) − p(G \\ explanation)   (high is good: removing the
                                                        explanation hurts)
              Fidelity− = p(G) − p(explanation only)   (low is good: the explanation
                                                        alone is sufficient)
          Reported both on probabilities and on accuracy (prediction == target), and
          against a random edge / node selection of the same size.

    Sanity checks (Adebayo et al. 2018, "Sanity Checks for Saliency Maps"):
        - Model-parameter randomization (cascading, top → bottom): heads, fc, then for
          the image branch cnn_fc and EfficientNetB2 blocks features.8 … features.0,
          and for the graph branch gcn2, gcn1. After each stage the explanation is
          recomputed and compared to the original one (Spearman rank correlation of
          signed and absolute attributions, top-k% IoU). A reliable explanation must
          CHANGE, i.e. similarity should drop towards 0.
        - Data (label) randomization: a copy of ImplantClassifier is trained from the
          same initialization as the original (ImageNet backbone) on the train split
          with permuted brand/diameter labels; explanations of that model are compared
          to the original's. Low similarity = the explanation depends on the
          input–label relationship the model learned.

(ii) STABILITY UNDER CONTROLLED PERTURBATIONS
    - Image: mild Gaussian noise, brightness and contrast changes, JPEG compression.
      No strong geometric perturbations (they would break the keypoint extraction).
      Perturbations are applied in [0, 1] pixel space on the masked crop and the
      implant mask is re-applied afterwards, mirroring the train-time preprocessing.
    - Graph: ±2–3 px noise on the keypoint coordinates (Gaussian, clipped), simulating
      the keypoint extractor's error. GNNExplainer is re-run with the same random seed
      so the variation comes from the input only; its intrinsic seed-to-seed
      variability on the unperturbed input is also reported as a reference.
    - ~20 perturbations per case. Each perturbed explanation is compared to the
      unperturbed one with:
          * Spearman correlation between the attribution rankings,
          * IoU / Jaccard of the top-k% pixels / edges / nodes,
          * local Lipschitz estimate  max_k ||E(x_k) − E(x)||₂ / ||x_k − x||₂
            (image maps divided by their max |value| — Grad-CAM → [0, 1], IG → [−1, 1]
            — before the norm; x in [0, 1] pixel space or in keypoint pixels).
      The fraction of perturbations that flip the model's prediction is also logged.

The explained class is fixed per case (the original model's prediction, or the
ground truth with --target true) and kept constant for every randomized model and
every perturbation, so that only the factor under study changes.

Grad-CAM / Grad-CAM++ are imported from step 7 and GNNExplainer / counterfactual node
importance from step 8, so the evaluated explanations are exactly the ones those
scripts produce. Integrated Gradients (Sundararajan et al. 2017) is implemented here
(black image baseline, midpoint Riemann sum) as the pixel-level gradient attribution
counterpart to Grad-CAM's coarse 9×9 map.

This script only *reads* the checkpoint trained by `fed_manual 1`'s `(6) train.py`.
Randomized models are deep copies; the label-randomized model is trained separately
and saved under outputs/evaluation/.

Outputs (outputs/evaluation/):
    fidelity_stability_results.json   mean ± SD per section/head/method/variant/metric/class
    fidelity_stability_summary.csv    same, as a flat table
    fidelity_stability_per_case.csv   every per-case value (for further statistics)
    label_randomized_model.pt (+ .json)
    figures/                          deletion/insertion curves, sanity-check plots,
                                      stability bar charts

Usage:
    python "(12) evaluate_explanations.py"
    python "(12) evaluate_explanations.py" --sections fidelity --image-methods gradcam,ig --neutral blur,gray
    python "(12) evaluate_explanations.py" --sections stability --n-perturb 20 --stability-max-per-class 5
    python "(12) evaluate_explanations.py" --sections sanity --skip-label-randomization
"""

import argparse
import copy
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as TVF
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
from scipy.stats import spearmanr

from common_explain import (
    SEED, HERE, NODE_KEYS, ActivationGrabber, ImplantProbeDataset,
    build_edge_index, build_label_maps_and_scaler, find_fed1_dir, find_train_csv,
    load_model, load_probe_df,
)


def _load_step_module(name: str, filename: str):
    """Imports a numbered script of this folder (its file name is not a valid module name)."""
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GC = _load_step_module("gradcam_explain", "(7) gradcam_explain.py")   # compute_cams
GE = _load_step_module("gnn_explain", "(8) gnn_explain.py")           # gnn_explain_instance, counterfactual_node_importance

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_NODES = len(NODE_KEYS)
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
EPS = 1e-8

IMAGE_METHODS = ("gradcam", "gradcampp", "ig")
GRAPH_METHODS = ("gnnexplainer", "counterfactual")
NEUTRALS = ("blur", "gray", "noise")
IMAGE_PERTURBATIONS = ("noise", "brightness", "contrast", "jpeg")


def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, x))


def denorm(img: torch.Tensor) -> torch.Tensor:
    return (img * IMG_STD.to(img.device) + IMG_MEAN.to(img.device)).clamp(0, 1)


def renorm(pix: torch.Tensor) -> torch.Tensor:
    return (pix - IMG_MEAN.to(pix.device)) / IMG_STD.to(pix.device)


# =============================================================================
#  RESULT RECORDER  (every per-case number goes through here)
# =============================================================================

class Recorder:
    def __init__(self):
        self.rows: List[Dict] = []

    def add(self, section, head, cls_name, method, variant, metric, case_id, value):
        self.rows.append(dict(section=section, head=head, cls=str(cls_name), method=method,
                              variant=str(variant), metric=metric, case=case_id,
                              value=float(value) if value is not None else np.nan))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """mean ± SD per class, plus an 'ALL' row pooling every case of the head."""
    keys = ["section", "head", "method", "variant", "metric"]

    def agg(group_keys):
        return (df.groupby(group_keys, sort=False)["value"]
                  .agg(mean="mean", std=lambda s: s.std(ddof=1), n="count").reset_index())

    per_class = agg(keys + ["cls"])
    overall = agg(keys)
    overall["cls"] = "ALL"
    return pd.concat([per_class, overall], ignore_index=True)


def summary_to_nested(summary: pd.DataFrame) -> Dict:
    nested: Dict = {}
    for r in summary.itertuples(index=False):
        d = nested.setdefault(r.section, {}).setdefault(r.head, {}).setdefault(r.method, {}) \
                  .setdefault(r.variant, {}).setdefault(r.metric, {})
        d[r.cls] = {"mean": None if pd.isna(r.mean) else float(r.mean),
                    "std": None if pd.isna(r.std) else float(r.std), "n": int(r.n)}
    return nested


# =============================================================================
#  SIMILARITY METRICS BETWEEN TWO EXPLANATIONS
# =============================================================================

def rank_corr(a: np.ndarray, b: np.ndarray, const_value: float = np.nan) -> float:
    """Spearman correlation; `const_value` if exactly one of the two is constant, nan if both."""
    a, b = np.ravel(a), np.ravel(b)
    ca, cb = np.ptp(a) == 0, np.ptp(b) == 0
    if ca and cb:
        return np.nan
    if ca or cb:
        return const_value
    rho = spearmanr(a, b)[0]
    return float(rho) if not np.isnan(rho) else const_value


def topk_mask(m: np.ndarray, frac: float) -> np.ndarray:
    flat = np.ravel(m)
    k = max(1, int(round(frac * flat.size)))
    mask = np.zeros(flat.size, dtype=bool)
    mask[np.argpartition(-flat, k - 1)[:k]] = True
    return mask


def topk_jaccard(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    ma, mb = topk_mask(a, frac), topk_mask(b, frac)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


def scale_explanation(m: np.ndarray, method: str) -> np.ndarray:
    """Comparable scale for Lipschitz: image maps are divided by their max |value|
    (Grad-CAM → [0, 1], signed IG → [−1, 1]); graph masks / Δprob are kept as-is."""
    if method in IMAGE_METHODS:
        peak = np.abs(m).max()
        return m / peak if peak > 0 else m
    return m


# =============================================================================
#  MODEL PIECES  (image branch with graph/scalar embeddings held fixed, graph branch
#  with edge weights + node removal, image held fixed)
# =============================================================================

@torch.no_grad()
def side_embeddings(model, node_feats, scalar_feat, device):
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


@torch.no_grad()
def image_target_probs(model, images, gnn_feat, scalar_emb, head, cls, batch_size) -> np.ndarray:
    probs = []
    for s in range(0, images.shape[0], batch_size):
        logits = image_branch_logits(model, images[s:s + batch_size], gnn_feat, scalar_emb, head)
        probs.append(TF.softmax(logits, dim=1)[:, cls])
    return torch.cat(probs).cpu().numpy()


@torch.no_grad()
def full_prediction(model, image, node_feats, scalar_feat, head, device) -> Tuple[np.ndarray, int]:
    gnn_feat, scalar_emb = side_embeddings(model, node_feats, scalar_feat, device)
    logits = image_branch_logits(model, image.unsqueeze(0).to(device), gnn_feat, scalar_emb, head)
    probs = TF.softmax(logits, dim=1)[0].cpu().numpy()
    return probs, int(probs.argmax())


@torch.no_grad()
def graph_target_prob(model, node_feats, edge_weight, node_keep, img_feat, scalar_emb,
                      head, cls, device) -> Tuple[float, int]:
    """Graph-branch forward where removed edges have weight 0 and removed nodes lose all
    incident edges and are excluded from the mean pool. Empty graph → zero embedding."""
    edge_index = build_edge_index(N_NODES, 1, device)
    keep = torch.as_tensor(node_keep, dtype=torch.float32, device=device)
    ew = torch.as_tensor(edge_weight, dtype=torch.float32, device=device) * keep[edge_index[0]] * keep[edge_index[1]]
    x = torch.relu(model.gcn1(node_feats, edge_index, edge_weight=ew))
    x = model.gcn2(x, edge_index, edge_weight=ew)
    keep_b = keep.bool()
    gnn_feat = x[keep_b].mean(0, keepdim=True) if keep_b.any() else torch.zeros(1, x.shape[1], device=device)
    out = model.fc(torch.cat([img_feat, gnn_feat, scalar_emb], dim=1))
    logits = model.brand_head(out) if head == "brand" else model.diameter_head(out)
    probs = TF.softmax(logits, dim=1)[0]
    return float(probs[cls].item()), int(probs.argmax().item())


# =============================================================================
#  IMAGE EXPLANATIONS  (Grad-CAM / Grad-CAM++ from step 7, IG implemented here)
# =============================================================================

def gradcam_maps(model, image, gnn_feat, scalar_emb, head, cls, device) -> Tuple[np.ndarray, np.ndarray]:
    grabber = ActivationGrabber(model.cnn_backbone.features)
    try:
        x = image.unsqueeze(0).to(device).requires_grad_(True)   # params are frozen
        logits = image_branch_logits(model, x, gnn_feat, scalar_emb, head)
        logits[0, cls].backward()
        A = grabber.activation.detach()
        G = grabber.activation.grad.detach()
    finally:
        grabber.remove()
    cam_gc, cam_gcpp = GC.compute_cams(A, G, image_size=tuple(image.shape[1:]))
    return cam_gc[0], cam_gcpp[0]


def integrated_gradients(model, image, gnn_feat, scalar_emb, head, cls, device,
                         steps: int, batch_size: int) -> np.ndarray:
    """IG w.r.t. the target logit, black (zero-pixel) baseline = the masked-out background
    value. Returns the channel-summed signed attribution map (H, W)."""
    x = image.to(device)
    x0 = renorm(torch.zeros_like(x))
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps
    total_grad = torch.zeros_like(x)
    for s in range(0, steps, batch_size):
        a = alphas[s:s + batch_size].view(-1, 1, 1, 1)
        xs = (x0 + a * (x - x0)).requires_grad_(True)
        logits = image_branch_logits(model, xs, gnn_feat, scalar_emb, head)
        grad, = torch.autograd.grad(logits[:, cls].sum(), xs)
        total_grad += grad.sum(0)
    attr = (x - x0) * total_grad / steps
    return attr.sum(0).detach().cpu().numpy()


def explain_image(model, methods, image, node_feats, scalar_feat, head, cls, device, args) -> Dict[str, np.ndarray]:
    gnn_feat, scalar_emb = side_embeddings(model, node_feats, scalar_feat, device)
    maps = {}
    if "gradcam" in methods or "gradcampp" in methods:
        gc, gcpp = gradcam_maps(model, image, gnn_feat, scalar_emb, head, cls, device)
        if "gradcam" in methods:
            maps["gradcam"] = gc
        if "gradcampp" in methods:
            maps["gradcampp"] = gcpp
    if "ig" in methods:
        maps["ig"] = integrated_gradients(model, image, gnn_feat, scalar_emb, head, cls, device,
                                          args.ig_steps, args.batch_size)
    return maps


# =============================================================================
#  GRAPH EXPLANATIONS  (from step 8)
# =============================================================================

def explain_graph(model, methods, image, node_feats, scalar_feat, head, cls, device, args,
                  seed: int = SEED) -> Dict[str, np.ndarray]:
    """Keys: gnnexplainer_edges (30), gnnexplainer_nodes (6), counterfactual_nodes (6)."""
    out = {}
    if "gnnexplainer" in methods:
        torch.manual_seed(seed)   # same mask init → differences come from the input/model only
        node_imp, edge_imp, _ = GE.gnn_explain_instance(model, image, node_feats, scalar_feat, head, cls,
                                                        device, args.gnn_steps, args.gnn_lr)
        out["gnnexplainer_edges"] = edge_imp
        out["gnnexplainer_nodes"] = node_imp
    if "counterfactual" in methods:
        cf_imp, _ = GE.counterfactual_node_importance(model, image, node_feats, scalar_feat, head, cls, device)
        out["counterfactual_nodes"] = cf_imp
    return out


# =============================================================================
#  (i-a) IMAGE FIDELITY — deletion / insertion vs. random order
# =============================================================================

def neutral_image(image: torch.Tensor, kind: str, seed: int) -> torch.Tensor:
    pix = denorm(image)
    if kind == "blur":
        neutral = TVF.gaussian_blur(pix.unsqueeze(0), kernel_size=51, sigma=15.0)[0]
    elif kind == "gray":
        neutral = torch.full_like(pix, 0.5)
    elif kind == "noise":
        g = torch.Generator().manual_seed(seed)
        neutral = (0.5 + 0.2 * torch.randn(pix.shape, generator=g)).clamp(0, 1).to(pix.device)
    else:
        raise ValueError(kind)
    return renorm(neutral)


def pixel_order(saliency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pixel indices sorted by attribution (desc), ties broken at random."""
    flat = np.ravel(saliency)
    return np.lexsort((rng.random(flat.size), -flat))


def deletion_insertion(model, image, neutral, order, n_steps, gnn_feat, scalar_emb, head, cls,
                       device, batch_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = image.shape[1:]
    n_pix = H * W
    fracs = np.linspace(0, 1, n_steps + 1)
    counts = torch.as_tensor(np.round(fracs * n_pix).astype(np.int64), device=device)
    rank = np.empty(n_pix, dtype=np.int64)
    rank[order] = np.arange(n_pix)
    rank_t = torch.as_tensor(rank, device=device).view(1, 1, H, W)
    x, x0 = image.to(device), neutral.to(device)

    del_probs, ins_probs = [], []
    for s in range(0, len(counts), batch_size):
        m = (rank_t < counts[s:s + batch_size].view(-1, 1, 1, 1)).float()   # 1 = already touched
        del_imgs = m * x0 + (1 - m) * x
        ins_imgs = m * x + (1 - m) * x0
        del_probs.append(image_target_probs(model, del_imgs, gnn_feat, scalar_emb, head, cls, batch_size))
        ins_probs.append(image_target_probs(model, ins_imgs, gnn_feat, scalar_emb, head, cls, batch_size))
    return fracs, np.concatenate(del_probs), np.concatenate(ins_probs)


def run_image_fidelity(model, case, maps, args, device, rec, curves, rng):
    gnn_feat, scalar_emb = side_embeddings(model, case["node"], case["scalar"], device)
    for neutral_kind in args.neutral:
        neutral = neutral_image(case["image"], neutral_kind, seed=SEED + case["idx"])
        # random-order baseline, shared by every method on this image/neutral
        rand_del, rand_ins = [], []
        for _ in range(args.n_random):
            order = rng.permutation(case["image"].shape[1] * case["image"].shape[2])
            fr, d, i = deletion_insertion(model, case["image"], neutral, order, args.curve_steps,
                                          gnn_feat, scalar_emb, case["head"], case["cls"], device, args.batch_size)
            rand_del.append(d); rand_ins.append(i)
        rand_del, rand_ins = np.mean(rand_del, 0), np.mean(rand_ins, 0)
        auc_rd, auc_ri = _auc(rand_del, fr), _auc(rand_ins, fr)

        for method, sal in maps.items():
            order = pixel_order(sal, rng)
            fr, d, i = deletion_insertion(model, case["image"], neutral, order, args.curve_steps,
                                          gnn_feat, scalar_emb, case["head"], case["cls"], device, args.batch_size)
            auc_d, auc_i = _auc(d, fr), _auc(i, fr)
            meta = ("fidelity_image", case["head"], case["cls_name"], method, neutral_kind)
            rec.add(*meta, "deletion_auc", case["id"], auc_d)
            rec.add(*meta, "deletion_auc_random", case["id"], auc_rd)
            rec.add(*meta, "deletion_gain_vs_random", case["id"], auc_rd - auc_d)
            rec.add(*meta, "insertion_auc", case["id"], auc_i)
            rec.add(*meta, "insertion_auc_random", case["id"], auc_ri)
            rec.add(*meta, "insertion_gain_vs_random", case["id"], auc_i - auc_ri)
            rec.add(*meta, "beats_random_deletion", case["id"], float(auc_d < auc_rd))
            rec.add(*meta, "beats_random_insertion", case["id"], float(auc_i > auc_ri))
            curves[("image", case["head"], method, neutral_kind, "deletion")].append((fr, d, rand_del))
            curves[("image", case["head"], method, neutral_kind, "insertion")].append((fr, i, rand_ins))


# =============================================================================
#  (i-b) GRAPH FIDELITY — edge/node deletion/insertion + Fidelity+/−
# =============================================================================

def _graph_eval(model, node_feats, img_feat, scalar_emb, case, device, edge_on=None, node_on=None):
    ew = np.ones(N_NODES * (N_NODES - 1)) if edge_on is None else edge_on.astype(float)
    nk = np.ones(N_NODES) if node_on is None else node_on.astype(float)
    return graph_target_prob(model, node_feats, ew, nk, img_feat, scalar_emb, case["head"], case["cls"], device)


def _graph_curves(model, node_feats, img_feat, scalar_emb, case, device, order, unit):
    n = len(order)
    del_p, ins_p = [], []
    for k in range(n + 1):
        sel = np.zeros(n, dtype=bool)
        sel[order[:k]] = True
        if unit == "edges":
            del_p.append(_graph_eval(model, node_feats, img_feat, scalar_emb, case, device, edge_on=~sel)[0])
            ins_p.append(_graph_eval(model, node_feats, img_feat, scalar_emb, case, device, edge_on=sel)[0])
        else:
            del_p.append(_graph_eval(model, node_feats, img_feat, scalar_emb, case, device, node_on=~sel)[0])
            ins_p.append(_graph_eval(model, node_feats, img_feat, scalar_emb, case, device, node_on=sel)[0])
    return np.linspace(0, 1, n + 1), np.array(del_p), np.array(ins_p)


def _fidelity(model, node_feats, img_feat, scalar_emb, case, device, selected, unit):
    """Returns (fid+_prob, fid−_prob, fid+_acc, fid−_acc) for a boolean selection."""
    p_full, pred_full = _graph_eval(model, node_feats, img_feat, scalar_emb, case, device)
    kw_rm = {"edge_on": ~selected} if unit == "edges" else {"node_on": ~selected}
    kw_only = {"edge_on": selected} if unit == "edges" else {"node_on": selected}
    p_rm, pred_rm = _graph_eval(model, node_feats, img_feat, scalar_emb, case, device, **kw_rm)
    p_only, pred_only = _graph_eval(model, node_feats, img_feat, scalar_emb, case, device, **kw_only)
    acc = lambda pred: float(pred == case["cls"])
    return p_full - p_rm, p_full - p_only, acc(pred_full) - acc(pred_rm), acc(pred_full) - acc(pred_only)


def run_graph_fidelity(model, case, gexp, args, device, rec, curves, rng):
    node_feats = case["node"].to(device)
    with torch.no_grad():
        img_feat = model.cnn_fc(model.cnn_backbone(case["image"].unsqueeze(0).to(device)))
        scalar_emb = model.scalar_fc(case["scalar"].unsqueeze(0).to(device))

    for key, imp in gexp.items():
        method, unit = key.rsplit("_", 1)          # e.g. gnnexplainer / edges
        n = len(imp)
        order = np.lexsort((rng.random(n), -imp))
        fr, d, i = _graph_curves(model, node_feats, img_feat, scalar_emb, case, device, order, unit)
        rand_d, rand_i = [], []
        for _ in range(args.n_random_graph):
            _, rd, ri = _graph_curves(model, node_feats, img_feat, scalar_emb, case, device, rng.permutation(n), unit)
            rand_d.append(rd); rand_i.append(ri)
        rand_d, rand_i = np.mean(rand_d, 0), np.mean(rand_i, 0)

        meta = ("fidelity_graph", case["head"], case["cls_name"], method, unit)
        auc_d, auc_i, auc_rd, auc_ri = _auc(d, fr), _auc(i, fr), _auc(rand_d, fr), _auc(rand_i, fr)
        rec.add(*meta, "deletion_auc", case["id"], auc_d)
        rec.add(*meta, "deletion_auc_random", case["id"], auc_rd)
        rec.add(*meta, "deletion_gain_vs_random", case["id"], auc_rd - auc_d)
        rec.add(*meta, "insertion_auc", case["id"], auc_i)
        rec.add(*meta, "insertion_auc_random", case["id"], auc_ri)
        rec.add(*meta, "insertion_gain_vs_random", case["id"], auc_i - auc_ri)
        curves[("graph", case["head"], method, unit, "deletion")].append((fr, d, rand_d))
        curves[("graph", case["head"], method, unit, "insertion")].append((fr, i, rand_i))

        for frac in args.fid_fracs:
            k = max(1, int(round(frac * n)))
            sel = np.zeros(n, dtype=bool); sel[order[:k]] = True
            fp, fm, fpa, fma = _fidelity(model, node_feats, img_feat, scalar_emb, case, device, sel, unit)
            rnd = []
            for _ in range(args.n_random_graph):
                rsel = np.zeros(n, dtype=bool); rsel[rng.permutation(n)[:k]] = True
                rnd.append(_fidelity(model, node_feats, img_feat, scalar_emb, case, device, rsel, unit))
            rfp, rfm, rfpa, rfma = np.mean(rnd, 0)
            tag = f"@{frac:g}(k={k})"
            rec.add(*meta, f"fidelity_plus_prob{tag}", case["id"], fp)
            rec.add(*meta, f"fidelity_minus_prob{tag}", case["id"], fm)
            rec.add(*meta, f"fidelity_plus_acc{tag}", case["id"], fpa)
            rec.add(*meta, f"fidelity_minus_acc{tag}", case["id"], fma)
            rec.add(*meta, f"fidelity_plus_prob_random{tag}", case["id"], rfp)
            rec.add(*meta, f"fidelity_minus_prob_random{tag}", case["id"], rfm)
            rec.add(*meta, f"fidelity_plus_acc_random{tag}", case["id"], rfpa)
            rec.add(*meta, f"fidelity_minus_acc_random{tag}", case["id"], rfma)


# =============================================================================
#  (i-c) SANITY CHECKS — model-parameter randomization + label randomization
# =============================================================================

def _reset_modules(modules, seed: int):
    torch.manual_seed(seed)
    for m in modules:
        for sub in m.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()


def build_cascading_randomizations(model, image_methods, graph_methods, seed: int) -> Tuple[List, List]:
    """Returns (image_cascade, graph_cascade): lists of (stage_name, randomized model copy),
    each stage randomizing one more block, from the output heads down to the input."""
    def frozen(m):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    top = [("heads", lambda m: [m.brand_head, m.diameter_head]), ("fc", lambda m: [m.fc])]
    image_chain = [("cnn_fc", lambda m: [m.cnn_fc])] + [
        (f"features.{i}", (lambda i: lambda m: [m.cnn_backbone.features[i]])(i))
        for i in reversed(range(len(model.cnn_backbone.features)))
    ]
    graph_chain = [("gcn2", lambda m: [m.gcn2]), ("gcn1", lambda m: [m.gcn1])]

    def cascade(chain):
        current = copy.deepcopy(model)
        stages = []
        for s, (name, getter) in enumerate(chain):
            _reset_modules(getter(current), seed + s)
            stages.append((name, frozen(copy.deepcopy(current))))
        return stages

    image_cascade = cascade(top + image_chain) if image_methods else []
    graph_cascade = cascade(top + graph_chain) if graph_methods else []
    return image_cascade, graph_cascade


def _compare(rec, section, case, method, variant, ref, other, topk_frac, signed=True):
    meta = (section, case["head"], case["cls_name"], method, variant)
    # a randomized explanation that collapses to a constant has changed completely → ρ = 0
    rec.add(*meta, "spearman", case["id"], rank_corr(ref, other, const_value=0.0))
    if signed:
        rec.add(*meta, "spearman_abs", case["id"], rank_corr(np.abs(ref), np.abs(other), const_value=0.0))
    rec.add(*meta, "topk_iou", case["id"], topk_jaccard(ref, other, topk_frac))


def run_sanity(model, case, ref_maps, ref_graph, cascades, label_model, args, device, rec, examples):
    image_cascade, graph_cascade = cascades
    signed_methods = {"ig", "counterfactual_nodes"}
    keep_example = case["head"] not in examples
    if keep_example:
        examples[case["head"]] = {"case": case["id"], "image": case["image"], "stages": [("original", ref_maps)]}

    for stage, m in image_cascade:
        maps = explain_image(m, list(ref_maps), case["image"], case["node"], case["scalar"],
                             case["head"], case["cls"], device, args)
        for method, sal in maps.items():
            _compare(rec, "sanity_model_randomization", case, method, stage, ref_maps[method], sal,
                     args.topk_frac, signed=method in signed_methods)
        if keep_example:
            examples[case["head"]]["stages"].append((stage, maps))

    for stage, m in graph_cascade:
        g = explain_graph(m, args.graph_methods, case["image"], case["node"], case["scalar"],
                          case["head"], case["cls"], device, args)
        for key, imp in g.items():
            _compare(rec, "sanity_model_randomization", case, key, stage, ref_graph[key], imp,
                     args.topk_frac, signed=key in signed_methods)

    if label_model is not None:
        if ref_maps:
            maps = explain_image(label_model, list(ref_maps), case["image"], case["node"], case["scalar"],
                                 case["head"], case["cls"], device, args)
            for method, sal in maps.items():
                _compare(rec, "sanity_label_randomization", case, method, "label_randomized",
                         ref_maps[method], sal, args.topk_frac, signed=method in signed_methods)
            if keep_example:
                examples[case["head"]]["stages"].append(("labels\nrandom", maps))
        if ref_graph:
            g = explain_graph(label_model, args.graph_methods, case["image"], case["node"], case["scalar"],
                              case["head"], case["cls"], device, args)
            for key, imp in g.items():
                _compare(rec, "sanity_label_randomization", case, key, "label_randomized",
                         ref_graph[key], imp, args.topk_frac, signed=key in signed_methods)


def get_label_randomized_model(args, fed1_dir, fed2_dir, brand_map, diameter_map, scaler,
                               probe_df, probe_split_dir, device, out_dir) -> Tuple[torch.nn.Module, Dict]:
    """Loads (or trains) a copy of ImplantClassifier trained on permuted labels."""
    sys.path.insert(0, str(fed1_dir))
    from models.ImplantClassifier import ImplantClassifier  # noqa: E402

    ckpt = Path(args.label_random_checkpoint) if args.label_random_checkpoint else out_dir / "label_randomized_model.pt"
    meta_path = ckpt.with_suffix(".json")
    torch.manual_seed(SEED)
    model = ImplantClassifier(n_brands=len(brand_map), n_diameters=len(diameter_map)).to(device)

    if ckpt.exists() and not args.retrain_label_random:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        stats = json.loads(meta_path.read_text()) if meta_path.exists() else {"checkpoint": str(ckpt)}
        print(f"[INFO] Label-randomized model loaded from {ckpt}")
    else:
        train_df, train_dir = load_probe_df(fed2_dir, "train", brand_map, diameter_map, scaler)
        perm = np.random.default_rng(SEED).permutation(len(train_df))
        train_df[["brand_idx", "diameter_idx"]] = train_df[["brand_idx", "diameter_idx"]].values[perm]
        loader = DataLoader(ImplantProbeDataset(train_df, train_dir / "cropped_images", train_dir / "masks"),
                            batch_size=args.batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=args.label_random_lr)
        print(f"[INFO] Training label-randomized model on {len(train_df)} train examples "
              f"for {args.label_random_epochs} epochs ...")
        history = []
        for epoch in range(args.label_random_epochs):
            model.train()
            correct_b = correct_d = total = 0
            loss_sum = 0.0
            for images, node_feats, scalar_feat, b_idx, d_idx, _ in loader:
                B = images.shape[0]
                images, node_feats, scalar_feat = images.to(device), node_feats.to(device), scalar_feat.to(device)
                b_idx, d_idx = b_idx.to(device), d_idx.to(device)
                batch_vec = torch.arange(B, device=device).repeat_interleave(N_NODES)
                out_b, out_d = model(images, node_feats.view(-1, 2), build_edge_index(N_NODES, B, device),
                                     batch_vec, scalar_feat)
                loss = TF.cross_entropy(out_b, b_idx) + TF.cross_entropy(out_d, d_idx)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item() * B
                correct_b += (out_b.argmax(1) == b_idx).sum().item()
                correct_d += (out_d.argmax(1) == d_idx).sum().item()
                total += B
            history.append({"epoch": epoch + 1, "loss": loss_sum / total,
                            "train_acc_brand_random_labels": correct_b / total,
                            "train_acc_diameter_random_labels": correct_d / total})
            print(f"    epoch {epoch + 1}: loss={loss_sum / total:.3f}  "
                  f"train acc (random labels) brand={correct_b / total:.2f} diam={correct_d / total:.2f}")
        torch.save(model.state_dict(), ckpt)
        stats = {"checkpoint": str(ckpt), "n_train": int(len(train_df)), "epochs": args.label_random_epochs,
                 "lr": args.label_random_lr, "history": history}

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # test accuracy against the TRUE labels (should be near chance)
    loader = DataLoader(ImplantProbeDataset(probe_df, probe_split_dir / "cropped_images", probe_split_dir / "masks"),
                        batch_size=args.batch_size, shuffle=False)
    cb = cd = tot = 0
    with torch.no_grad():
        for images, node_feats, scalar_feat, b_idx, d_idx, _ in loader:
            B = images.shape[0]
            batch_vec = torch.arange(B, device=device).repeat_interleave(N_NODES)
            out_b, out_d = model(images.to(device), node_feats.to(device).view(-1, 2),
                                 build_edge_index(N_NODES, B, device), batch_vec, scalar_feat.to(device))
            cb += (out_b.argmax(1).cpu() == b_idx).sum().item()
            cd += (out_d.argmax(1).cpu() == d_idx).sum().item()
            tot += B
    stats["test_acc_brand_true_labels"] = cb / max(tot, 1)
    stats["test_acc_diameter_true_labels"] = cd / max(tot, 1)
    meta_path.write_text(json.dumps(stats, indent=2))
    print(f"[INFO] Label-randomized model test acc (true labels): brand={stats['test_acc_brand_true_labels']:.2f} "
          f"diameter={stats['test_acc_diameter_true_labels']:.2f}")
    return model, stats


# =============================================================================
#  (ii) STABILITY UNDER CONTROLLED PERTURBATIONS
# =============================================================================

def perturb_image(image: torch.Tensor, kind: str, rng: np.random.Generator, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (perturbed normalized image, perturbed pixels in [0, 1]). Implant mask re-applied."""
    pix = denorm(image.cpu())
    implant = (pix.sum(0, keepdim=True) > 0).float()
    if kind == "noise":
        out = pix + torch.from_numpy(rng.normal(0, args.noise_sigma, pix.shape)).float()
    elif kind == "brightness":
        out = TVF.adjust_brightness(pix, float(rng.uniform(1 - args.brightness, 1 + args.brightness)))
    elif kind == "contrast":
        out = TVF.adjust_contrast(pix, float(rng.uniform(1 - args.contrast, 1 + args.contrast)))
    elif kind == "jpeg":
        q = int(rng.integers(args.jpeg_quality[0], args.jpeg_quality[1] + 1))
        arr = (pix.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out = torch.from_numpy(dec).permute(2, 0, 1).float() / 255.0
    else:
        raise ValueError(kind)
    out = out.clamp(0, 1) * implant
    return renorm(out), out


def perturb_keypoints(node_feats: torch.Tensor, rng: np.random.Generator, args) -> torch.Tensor:
    noise = np.clip(rng.normal(0, args.kp_sigma, node_feats.shape), -args.kp_max, args.kp_max)
    return node_feats + torch.from_numpy(noise).float()


def _record_stability(rec, section, case, method, variant, ref, others, deltas, preds, args):
    meta = (section, case["head"], case["cls_name"], method, variant)
    rhos = [rank_corr(ref, o) for o in others]
    jacc = [topk_jaccard(ref, o, args.topk_frac) for o in others]
    ref_s = scale_explanation(ref, method)
    lips = [np.linalg.norm(scale_explanation(o, method) - ref_s) / (d + EPS) for o, d in zip(others, deltas) if d > 0]
    rec.add(*meta, "spearman", case["id"], np.nanmean(rhos) if not np.all(np.isnan(rhos)) else np.nan)
    rec.add(*meta, "topk_jaccard", case["id"], np.mean(jacc))
    rec.add(*meta, "lipschitz_local", case["id"], np.max(lips) if lips else np.nan)
    if preds is not None:
        rec.add(*meta, "prediction_flip_rate", case["id"], np.mean([p != case["pred"] for p in preds]))


def run_stability(model, case, ref_maps, ref_graph, args, device, rec, rng):
    # ---- image branch ----
    if ref_maps:
        pix0 = denorm(case["image"].cpu())
        for kind in args.image_perturbations:
            per_method = defaultdict(list)
            deltas, preds = [], []
            for _ in range(args.n_perturb):
                img_p, pix_p = perturb_image(case["image"], kind, rng, args)
                deltas.append(float(torch.linalg.vector_norm(pix_p - pix0)))
                preds.append(full_prediction(model, img_p, case["node"], case["scalar"], case["head"], device)[1])
                for method, sal in explain_image(model, list(ref_maps), img_p, case["node"], case["scalar"],
                                                 case["head"], case["cls"], device, args).items():
                    per_method[method].append(sal)
            for method, sals in per_method.items():
                _record_stability(rec, "stability_image", case, method, kind, ref_maps[method], sals,
                                  deltas, preds, args)

    # ---- graph branch ----
    if ref_graph:
        per_key = defaultdict(list)
        deltas, preds = [], []
        for _ in range(args.n_perturb):
            node_p = perturb_keypoints(case["node"], rng, args)
            deltas.append(float(torch.linalg.vector_norm(node_p - case["node"])))
            preds.append(full_prediction(model, case["image"], node_p, case["scalar"], case["head"], device)[1])
            for key, imp in explain_graph(model, args.graph_methods, case["image"], node_p, case["scalar"],
                                          case["head"], case["cls"], device, args).items():
                per_key[key].append(imp)
        for key, imps in per_key.items():
            _record_stability(rec, "stability_graph", case, key, f"keypoint_noise_{args.kp_max:g}px",
                              ref_graph[key], imps, deltas, preds, args)

        # intrinsic GNNExplainer variability: same input, different mask initializations
        if "gnnexplainer" in args.graph_methods and args.intrinsic_repeats > 0:
            reps = defaultdict(list)
            for r in range(args.intrinsic_repeats):
                g = explain_graph(model, ["gnnexplainer"], case["image"], case["node"], case["scalar"],
                                  case["head"], case["cls"], device, args, seed=SEED + 1000 + r)
                for key, imp in g.items():
                    reps[key].append(imp)
            for key, imps in reps.items():
                meta = ("stability_graph", case["head"], case["cls_name"], key, "intrinsic_seed_only")
                rec.add(*meta, "spearman", case["id"], np.nanmean([rank_corr(ref_graph[key], o) for o in imps]))
                rec.add(*meta, "topk_jaccard", case["id"],
                        np.mean([topk_jaccard(ref_graph[key], o, args.topk_frac) for o in imps]))


# =============================================================================
#  FIGURES
# =============================================================================

def plot_curves(curves, fig_dir: Path):
    groups = defaultdict(dict)
    for (branch, head, method, variant, kind), items in curves.items():
        groups[(branch, head, method, variant)][kind] = items
    for (branch, head, method, variant), kinds in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, kind in zip(axes, ("deletion", "insertion")):
            if kind not in kinds:
                continue
            fr = kinds[kind][0][0]
            y = np.stack([c[1] for c in kinds[kind]])
            yr = np.stack([c[2] for c in kinds[kind]])
            for arr, label, color in ((y, method, "#1f6feb"), (yr, "random order", "#8b949e")):
                m, s = arr.mean(0), arr.std(0)
                ax.plot(fr, m, label=f"{label} (AUC={_auc(m, fr):.3f})", color=color)
                ax.fill_between(fr, m - s, m + s, color=color, alpha=0.15)
            unit = "pixels" if branch == "image" else variant
            ax.set_xlabel(f"fraction of {unit} {'removed' if kind == 'deletion' else 'inserted'}")
            ax.set_ylabel("target-class probability")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{kind.capitalize()} (n={len(y)})", fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(f"{branch} branch — {head} — {method} — {variant}", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"curves_{branch}_{head}_{_safe(method)}_{_safe(variant)}.png", dpi=130)
        plt.close(fig)


def plot_sanity(summary: pd.DataFrame, fig_dir: Path):
    df = summary[(summary.section == "sanity_model_randomization") & (summary.cls == "ALL")
                 & (summary.metric == "spearman")]
    for head, dh in df.groupby("head"):
        # image and graph methods have different cascades, so they get separate figures
        for branch, dfam in (("image", dh[dh.method.isin(IMAGE_METHODS)]),
                             ("graph", dh[~dh.method.isin(IMAGE_METHODS)])):
            if dfam.empty:
                continue
            fig, ax = plt.subplots(figsize=(10 if branch == "image" else 6, 4))
            for method, dm in dfam.groupby("method", sort=False):
                ax.errorbar(range(len(dm)), dm["mean"], yerr=dm["std"].fillna(0), marker="o", capsize=3, label=method)
                ax.set_xticks(range(len(dm)))
                ax.set_xticklabels(dm["variant"], rotation=45, ha="right", fontsize=7)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("Spearman ρ (original vs randomized)")
            ax.set_xlabel("cascading randomization (top → bottom)")
            ax.set_title(f"Model-parameter randomization sanity check — {branch} branch — {head}", fontsize=10)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / f"sanity_cascading_{branch}_{head}.png", dpi=130)
            plt.close(fig)


def plot_sanity_examples(examples, fig_dir: Path):
    for head, ex in examples.items():
        stages = ex["stages"]
        methods = list(stages[0][1].keys())
        if not methods:
            continue
        fig, axes = plt.subplots(len(methods), len(stages) + 1,
                                 figsize=(1.6 * (len(stages) + 1), 1.8 * len(methods)), squeeze=False)
        base = denorm(ex["image"].cpu()).permute(1, 2, 0).numpy()
        for r, method in enumerate(methods):
            axes[r, 0].imshow(base)
            axes[r, 0].set_ylabel(method, fontsize=8)
            for c, (stage, maps) in enumerate(stages, start=1):
                m = maps[method]
                if method == "ig":
                    m = np.abs(m) / (np.abs(m).max() + EPS)
                axes[r, c].imshow(m, cmap="jet", vmin=0, vmax=1)
                if r == 0:
                    axes[r, c].set_title(stage, fontsize=6)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Sanity check example — {head} — {ex['case']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"sanity_example_{head}.png", dpi=130)
        plt.close(fig)


def plot_stability(summary: pd.DataFrame, fig_dir: Path):
    df = summary[summary.section.str.startswith("stability") & (summary.cls == "ALL")]
    for head, dh in df.groupby("head"):
        metrics = ["spearman", "topk_jaccard", "lipschitz_local"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        for ax, metric in zip(axes, metrics):
            d = dh[dh.metric == metric]
            labels = [f"{m}\n{v}" for m, v in zip(d.method, d.variant)]
            ax.bar(range(len(d)), d["mean"], yerr=d["std"].fillna(0), capsize=3, color="#2a9d8f")
            ax.set_xticks(range(len(d)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
            ax.set_title(metric, fontsize=9)
        fig.suptitle(f"Explanation stability under perturbations — {head} (mean ± SD over cases)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"stability_{head}.png", dpi=130)
        plt.close(fig)


def print_overview(summary: pd.DataFrame):
    key_metrics = {"deletion_auc", "deletion_auc_random", "insertion_auc", "insertion_auc_random",
                   "spearman", "topk_iou", "topk_jaccard", "lipschitz_local"}
    df = summary[(summary.cls == "ALL") & (summary.metric.isin(key_metrics)
                                          | summary.metric.str.startswith("fidelity_"))]
    for (section, head), d in df.groupby(["section", "head"], sort=False):
        print(f"\n=== {section} — {head} ===")
        for r in d.itertuples(index=False):
            print(f"  {r.method:22s} {r.variant:28s} {r.metric:42s} {r.mean:7.3f} ± {r.std:.3f}  (n={r.n})")


# =============================================================================
#  MAIN
# =============================================================================

def _csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Fidelity, sanity-check and stability evaluation of the explainability layer.")
    p.add_argument("--split", default="test")
    p.add_argument("--heads", default="brand,diameter", help="Comma-separated subset of {brand,diameter}.")
    p.add_argument("--target", choices=["predicted", "true"], default="predicted")
    p.add_argument("--model", default=None, help="Specific checkpoint (.pt). Default: latest in fed_manual 1.")
    p.add_argument("--device", default=None)
    p.add_argument("--sections", default="fidelity,sanity,stability",
                   help="Comma-separated subset of {fidelity,sanity,stability}.")
    p.add_argument("--image-methods", default="gradcam,gradcampp,ig", help=f"Subset of {IMAGE_METHODS}.")
    p.add_argument("--graph-methods", default="gnnexplainer,counterfactual", help=f"Subset of {GRAPH_METHODS}.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-examples-per-class", type=int, default=5)
    p.add_argument("--max-examples-per-class", type=int, default=10, help="Cases per class for fidelity.")
    p.add_argument("--sanity-max-per-class", type=int, default=3, help="Cases per class for sanity checks.")
    p.add_argument("--stability-max-per-class", type=int, default=5, help="Cases per class for stability.")
    p.add_argument("--topk-frac", type=float, default=0.2, help="k for top-k%% IoU/Jaccard.")
    # explainers
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--gnn-steps", type=int, default=150)
    p.add_argument("--gnn-lr", type=float, default=0.05)
    # fidelity
    p.add_argument("--neutral", default="blur", help=f"Comma-separated subset of {NEUTRALS}.")
    p.add_argument("--curve-steps", type=int, default=50, help="Points on each pixel deletion/insertion curve.")
    p.add_argument("--n-random", type=int, default=3, help="Random pixel orders averaged for the image baseline.")
    p.add_argument("--n-random-graph", type=int, default=20, help="Random edge/node orders for the graph baseline.")
    p.add_argument("--fid-fracs", default="0.1,0.2,0.3,0.5", help="Explanation sizes (fraction of edges/nodes) for Fidelity+/−.")
    # sanity
    p.add_argument("--skip-label-randomization", action="store_true")
    p.add_argument("--label-random-checkpoint", default=None,
                   help="Existing label-randomized checkpoint; default outputs/evaluation/label_randomized_model.pt.")
    p.add_argument("--retrain-label-random", action="store_true")
    p.add_argument("--label-random-epochs", type=int, default=15)
    p.add_argument("--label-random-lr", type=float, default=1e-4)
    # stability
    p.add_argument("--n-perturb", type=int, default=20)
    p.add_argument("--image-perturbations", default="noise,brightness,contrast,jpeg",
                   help=f"Comma-separated subset of {IMAGE_PERTURBATIONS}.")
    p.add_argument("--noise-sigma", type=float, default=0.02, help="Gaussian noise SD in [0,1] pixel units.")
    p.add_argument("--brightness", type=float, default=0.1, help="Brightness factor ~ U(1-b, 1+b).")
    p.add_argument("--contrast", type=float, default=0.1, help="Contrast factor ~ U(1-c, 1+c).")
    p.add_argument("--jpeg-quality", default="70,95", help="JPEG quality range (min,max).")
    p.add_argument("--kp-sigma", type=float, default=2.0, help="Keypoint noise SD (px).")
    p.add_argument("--kp-max", type=float, default=3.0, help="Keypoint noise clip (±px).")
    p.add_argument("--intrinsic-repeats", type=int, default=5,
                   help="GNNExplainer re-runs with other seeds on the clean input (reference variability).")
    args = p.parse_args()

    args.sections = set(_csv(args.sections))
    args.heads = _csv(args.heads)
    args.image_methods = [m for m in _csv(args.image_methods) if m]
    args.graph_methods = [m for m in _csv(args.graph_methods) if m]
    args.neutral = _csv(args.neutral)
    args.image_perturbations = _csv(args.image_perturbations)
    args.fid_fracs = [float(x) for x in _csv(args.fid_fracs)]
    args.jpeg_quality = [int(x) for x in _csv(args.jpeg_quality)]
    for name, values, allowed in (("--image-methods", args.image_methods, IMAGE_METHODS),
                                  ("--graph-methods", args.graph_methods, GRAPH_METHODS),
                                  ("--neutral", args.neutral, NEUTRALS),
                                  ("--image-perturbations", args.image_perturbations, IMAGE_PERTURBATIONS),
                                  ("--sections", args.sections, ("fidelity", "sanity", "stability"))):
        bad = set(values) - set(allowed)
        if bad:
            p.error(f"{name}: unknown value(s) {sorted(bad)}; allowed: {allowed}")
    return args


def main():
    args = parse_args()
    t0 = time.time()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    fed1_dir = find_fed1_dir()
    fed2_dir = HERE
    brand_map, diameter_map, scaler = build_label_maps_and_scaler(find_train_csv(fed1_dir))
    model, ckpt_path = load_model(fed1_dir, brand_map, diameter_map, device, args.model)
    for prm in model.parameters():
        prm.requires_grad_(False)
    print(f"[INFO] Checkpoint: {ckpt_path}")

    probe_df, probe_split_dir = load_probe_df(fed2_dir, args.split, brand_map, diameter_map, scaler)
    print(f"[INFO] Probe split '{args.split}': {len(probe_df)} usable examples.")

    out_dir = fed2_dir / "outputs" / "evaluation"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cascades, label_model, label_stats = ([], []), None, None
    if "sanity" in args.sections:
        print("[INFO] Building cascading parameter randomizations ...")
        cascades = build_cascading_randomizations(model, args.image_methods, args.graph_methods, SEED)
        if not args.skip_label_randomization:
            label_model, label_stats = get_label_randomized_model(
                args, fed1_dir, fed2_dir, brand_map, diameter_map, scaler, probe_df, probe_split_dir, device, out_dir)

    rec = Recorder()
    curves = defaultdict(list)
    sanity_examples: Dict = {}
    rng = np.random.default_rng(SEED)

    caps = {"fidelity": args.max_examples_per_class, "sanity": args.sanity_max_per_class,
            "stability": args.stability_max_per_class}
    max_cases = max(caps[s] for s in args.sections)

    for head, label_map, idx_col in (("brand", brand_map, "brand_idx"), ("diameter", diameter_map, "diameter_idx")):
        if head not in args.heads:
            continue
        for cls_name, cls_idx in label_map.items():
            subset = probe_df[probe_df[idx_col] == cls_idx]
            if len(subset) < args.min_examples_per_class:
                continue
            subset = subset.sample(min(len(subset), max_cases), random_state=SEED)
            ds = ImplantProbeDataset(subset, probe_split_dir / "cropped_images", probe_split_dir / "masks")
            print(f"[{head}={cls_name}] {len(ds)} cases")

            for i in range(len(ds)):
                image, node_feats, scalar_feat, b_idx, d_idx, image_name = ds[i]
                true_idx = b_idx if head == "brand" else d_idx
                probs, pred = full_prediction(model, image, node_feats, scalar_feat, head, device)
                target = true_idx if args.target == "true" else pred
                case = dict(id=str(image_name), idx=i, head=head, cls_name=cls_name, cls=int(target), pred=pred,
                            image=image, node=node_feats, scalar=scalar_feat)

                ref_maps = explain_image(model, args.image_methods, image, node_feats, scalar_feat,
                                         head, target, device, args) if args.image_methods else {}
                ref_graph = explain_graph(model, args.graph_methods, image, node_feats, scalar_feat,
                                          head, target, device, args) if args.graph_methods else {}

                if "fidelity" in args.sections and i < caps["fidelity"]:
                    if ref_maps:
                        run_image_fidelity(model, case, ref_maps, args, device, rec, curves, rng)
                    if ref_graph:
                        run_graph_fidelity(model, case, ref_graph, args, device, rec, curves, rng)
                if "sanity" in args.sections and i < caps["sanity"]:
                    run_sanity(model, case, ref_maps, ref_graph, cascades, label_model, args, device,
                               rec, sanity_examples)
                if "stability" in args.sections and i < caps["stability"]:
                    run_stability(model, case, ref_maps, ref_graph, args, device, rec, rng)
                print(f"    case {i + 1}/{len(ds)} ({image_name}) done  [{time.time() - t0:.0f}s]")

    df = rec.frame()
    if df.empty:
        print("[WARN] No results — check --min-examples-per-class and that the split has data.")
        return
    summary = summarize(df)
    df.to_csv(out_dir / "fidelity_stability_per_case.csv", index=False)
    summary.to_csv(out_dir / "fidelity_stability_summary.csv", index=False)

    results = {
        "config": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()},
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_probe_examples": int(len(probe_df)),
        "label_randomized_model": label_stats,
        "interpretation": {
            "deletion_auc": "lower is better; compare with deletion_auc_random",
            "insertion_auc": "higher is better; compare with insertion_auc_random",
            "*_gain_vs_random": "> 0 means the explanation beats a random ordering",
            "fidelity_plus_*": "higher is better (removing the explanation hurts the prediction)",
            "fidelity_minus_*": "lower is better (the explanation alone preserves the prediction)",
            "sanity_*: spearman / topk_iou": "should be LOW (explanation must change when the model/labels are randomized)",
            "stability_*: spearman / topk_jaccard": "higher = more stable",
            "stability_*: lipschitz_local": "lower = more stable",
        },
        "results": summary_to_nested(summary),
    }
    with open(out_dir / "fidelity_stability_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)

    plot_curves(curves, fig_dir)
    if "sanity" in args.sections:
        plot_sanity(summary, fig_dir)
        plot_sanity_examples(sanity_examples, fig_dir)
    if "stability" in args.sections:
        plot_stability(summary, fig_dir)

    print_overview(summary)
    print(f"\n[INFO] Saved {out_dir / 'fidelity_stability_results.json'}")
    print(f"[INFO] Per-case values : {out_dir / 'fidelity_stability_per_case.csv'}")
    print(f"[INFO] Figures         : {fig_dir}")
    print(f"[INFO] Total time      : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
