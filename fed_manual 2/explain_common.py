"""
explain_common.py — Shared loading/preprocessing utilities for the explainability
scripts in this folder (Grad-CAM, GNNExplainer, SHAP, TCAV, CBM, ...).

Every explanation technique needs the same three things before it can do anything
technique-specific:
    1. The trained ImplantClassifier checkpoint from `fed_manual 1` (read-only).
    2. The brand/diameter label maps + scalar-feature StandardScaler, refit exactly
       as `fed_manual 1`'s `(6) train.py` did, so class indices line up with the
       loaded checkpoint's output heads.
    3. A `Dataset` that turns a `features.csv` row back into the five tensors the
       multimodal model expects (image, node_feats, edge_index, batch, scalar_feat).

This module holds only that shared plumbing — no explanation logic lives here.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch_geometric.utils import dense_to_sparse

from PIL import Image
from sklearn.preprocessing import StandardScaler

SEED = 1225

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

HERE = Path(__file__).parent.resolve()


# =============================================================================
#  PATH DISCOVERY
# =============================================================================

def find_fed1_dir() -> Path:
    """Locates the sibling `fed_manual 1` folder (trained models + train CSV)."""
    env = os.environ.get("FED_MANUAL_1_DIR")
    if env:
        return Path(env).resolve()
    repo_root = HERE.parent
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


def find_train_csv(fed1_dir: Path) -> Path:
    train_csv = fed1_dir / "outputs" / "dataset_split" / "train" / "features_augmented.csv"
    if not train_csv.exists():
        train_csv = fed1_dir / "outputs" / "dataset_split" / "train" / "features.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Could not find features_augmented.csv or features.csv under "
            f"{train_csv.parent}. Run fed_manual 1's pipeline first."
        )
    return train_csv


# =============================================================================
#  LABEL MAPS / SCALER / MODEL  (must mirror fed_manual 1's train.py exactly)
# =============================================================================

def build_label_maps_and_scaler(train_csv: Path) -> Tuple[Dict, Dict, StandardScaler]:
    df = pd.read_csv(train_csv)
    scaler = StandardScaler()
    scaler.fit(df[SCALAR_COLS])
    brands = sorted(df["brand"].unique())
    diameters = sorted(df["implant_diameter_mm"].unique())
    brand_map = {b: i for i, b in enumerate(brands)}
    diameter_map = {d: i for i, d in enumerate(diameters)}
    return brand_map, diameter_map, scaler


def load_model(fed1_dir: Path, brand_map: Dict, diameter_map: Dict, device,
               model_path: Optional[str] = None):
    """Instantiates ImplantClassifier and loads the trained (read-only) checkpoint."""
    sys.path.insert(0, str(fed1_dir))
    from models.ImplantClassifier import ImplantClassifier  # noqa: E402

    ckpt_path = Path(model_path) if model_path else find_latest_checkpoint(fed1_dir / "outputs" / "models")
    model = ImplantClassifier(n_brands=len(brand_map), n_diameters=len(diameter_map)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, ckpt_path


def load_probe_df(fed2_dir: Path, split: str, brand_map: Dict, diameter_map: Dict,
                   scaler: StandardScaler) -> Tuple[pd.DataFrame, Path]:
    """Loads a split's features.csv, keeps only rows usable by the loaded checkpoint."""
    split_dir = fed2_dir / "outputs" / "dataset_split" / split
    probe_csv = split_dir / "features.csv"
    if not probe_csv.exists():
        raise FileNotFoundError(f"{probe_csv} not found. Run this folder's '(4) keypoints_extraction.py' first.")
    df = pd.read_csv(probe_csv)

    existing = {p.stem for p in (split_dir / "cropped_images").glob("*.png")}
    df = df[df["image_name"].isin(existing)]
    df = df[df["brand"].isin(brand_map) & df["implant_diameter_mm"].isin(diameter_map)]
    df = df.dropna(subset=SCALAR_COLS).reset_index(drop=True)
    df[SCALAR_COLS] = scaler.transform(df[SCALAR_COLS])
    df["brand_idx"] = df["brand"].map(brand_map)
    df["diameter_idx"] = df["implant_diameter_mm"].map(diameter_map)
    return df, split_dir


def parse_node(val):
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
#  PROBE DATASET  (turns a features.csv row into the model's five input tensors)
# =============================================================================

class ImplantProbeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, mask_dir: Path):
        self.data = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = str(row["image_name"])
        img_path = self.img_dir / f"{image_name}.png"
        mask_path = self.mask_dir / f"{image_name}_seg.png"
        image = IMAGE_TRANSFORM(load_masked_image(img_path, mask_path))
        node_feats = torch.tensor([parse_node(row[k]) for k in NODE_KEYS], dtype=torch.float32)
        scalar_feat = torch.tensor([row[c] for c in SCALAR_COLS], dtype=torch.float32)
        return image, node_feats, scalar_feat, int(row["brand_idx"]), int(row["diameter_idx"]), image_name


# =============================================================================
#  GRADIENT HOOK
# =============================================================================

class ActivationGrabber:
    """Hooks `module`, keeps its forward output and retains its gradient for backprop."""

    def __init__(self, module: torch.nn.Module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        out.retain_grad()
        self.activation = out

    def remove(self):
        self.handle.remove()
