# 📁 `fed_manual 2` — Explainability Layer for the Dental Implant Identification Model

`fed_manual 2` is a parallel folder to [`fed_manual 1`](../fed_manual%201), which contains the pipeline used to build the base multimodal prediction model (UNet segmentation + EfficientNet/ResNet + GNN classifier) as part of Paula Gisbert's TFM.

**This folder does not rebuild that base model.** It adds a post-hoc and concept-based **explainability layer** on top of the already-trained models from `fed_manual 1`, plus the quantitative and expert-based evaluation of that layer, as part of this author's own TFM: *"Explainable AI layer for multimodal dental implant identification model."*

Some scripts here are copies of `fed_manual 1` scripts, lightly adapted (e.g. `keypoints_extraction.py`, to additionally persist clinical concepts and build the concept dataset TCAV needs). Others are entirely new, written for this TFM. The distinction is called out per file below.

---

## Relationship to `fed_manual 1`

```
fed_manual 1/  →  produces the trained models this folder explains
    outputs/models/        UNet checkpoint, ImplantClassifier checkpoint
    outputs/dataset_split/ train/val/test images, masks, features.csv

fed_manual 2/  →  loads those models + that data, and explains + evaluates them
```

Nothing in `fed_manual 2` retrains the UNet or the `ImplantClassifier` from scratch. Where a technique needs access to intermediate activations (Grad-CAM, TCAV, GNNExplainer), the trained model is loaded read-only and never modified in place.

---

## Explainability technique per branch

The base model is multimodal (image branch, graph branch, scalar-feature branch, plus a derived concept layer). Each branch gets the technique whose documented limitations in the literature don't apply to it, and avoids the technique whose limitations do:

| Branch | Primary technique | Secondary / comparison | Library |
|---|---|---|---|
| Image (EfficientNetB2) | Grad-CAM | Grad-CAM++ | `captum` |
| Graph (6 keypoints, GNN) | GNNExplainer | Counterfactual perturbation of node coordinates | `torch_geometric` |
| Scalar features (7 features) | SHAP | Permutation importance | `shap` |
| Clinical concepts (tipo_implante, thread angle) | TCAV | Partial concept bottleneck (comparison) | custom |

---

## Folder contents

| # | Script | Status | Purpose |
|---|---|---|---|
| 0 | `(0) run_all.py` | Identical to `fed_manual 1` | Master pipeline runner, copied over unchanged so steps 1–4 can be re-run standalone inside this folder if needed. |
| 1 | `(1) prepare_dataset.py` | Identical to `fed_manual 1` | Merges CVAT COCO zip exports into a single unified dataset. Not modified — included so the full data pipeline can run from within `fed_manual 2` without depending on `fed_manual 1`'s copy. |
| 2 | `(2) dataset_analysis.py` | Identical to `fed_manual 1` | Dataset statistics, cleanup, per-category augmentation factors. Not modified. |
| 3 | `(3) split.py` | Identical to `fed_manual 1` | Train/val/test (70/15/15) split. Not modified. |
| 4 | `(4) keypoints_extraction.py` | Adapted from `fed_manual 1` | Same anatomical keypoint extraction as the original, plus: persists `tipo_implante` (U/Recto) as a CSV column, and adds `build_concept_dataset()` — exports clean (unannotated) crops into `concepts/<concept_name>/{positive,negative}/` per split, for use as TCAV probe sets. Concepts included so far: `tipo_implante_U`, `tipo_implante_Recto`, `steep_thread_angle`. |
| 5 | `(5) augmentation.py` | Identical to `fed_manual 1` | Two-stage crop + intensity augmentation. Not modified — confirmed it ignores the new `tipo_implante` column and the `concepts/` folder added in step 4. |
| 6 | `(6) train.py` | Identical to `fed_manual 1` | Trains the base UNet + ImplantClassifier. Kept here only so the base models can be reproduced from this folder if needed; not re-run in practice since `fed_manual 1`'s trained checkpoints are reused as-is. |
| 7 | `(7) gradcam_explain.py` | Planned | Grad-CAM / Integrated Gradients over the EfficientNetB2 branch (`captum`). Produces saliency maps per test image and aggregate visualizations. |
| 8 | `(8) gnn_explain.py` | Planned | GNNExplainer over the graph branch (6-keypoint graph). Identifies which keypoints/edges most influence brand/diameter predictions; complements with counterfactual perturbation of node coordinates. |
| 9 | `(9) shap_explain.py` | Planned | SHAP over the 7 scalar features (`distance_cm`, `pixel_per_cm`, `bbox_width`, `bbox_height`, `implant_bbox_ratio`, `angle_left_valley`, `angle_right_valley`). |
| 10 | `(10) tcav_analysis.py` | Planned | Trains CAVs from the `concepts/` folders built in step 4, computes directional derivatives (TCAV scores) against the trained `ImplantClassifier`. |
| 11 | `(11) cbm_partial.py` | Planned | Partial concept-bottleneck head trained alongside brand/diameter heads (shared trunk, no full model redesign), for direct comparison against TCAV on the same concept set. |
| 12 | `(12) evaluate_explanations.py` | Planned | Quantitative reliability evaluation across all techniques above: insertion/deletion, fidelity+/-, sanity checks, stability under input perturbations. Run on the test set. |
| 13 | `(13) expert_study/` | Planned | Materials and analysis for the expert evaluation study: final questionnaire, case selection protocol, session recording/analysis scripts. Feeds the discussion section on clinical usefulness and trust (RQ3: problematic-case detection). |

---

## Typical usage

```bash
# Step 4 (adapted): re-run keypoint extraction + build the concept dataset
python "(4) keypoints_extraction.py"

# Steps 7-9: per-branch attribution methods (require step 4 output + fed_manual 1 trained models)
python "(7) gradcam_explain.py"
python "(8) gnn_explain.py"
python "(9) shap_explain.py"

# Step 10: TCAV (requires the concepts/ folders from step 4)
python "(10) tcav_analysis.py"

# Step 11: partial CBM comparison
python "(11) cbm_partial.py"

# Step 12: quantitative evaluation of all of the above
python "(12) evaluate_explanations.py"
```

## Output structure (target)

```
outputs/
├── dataset_split/
│   ├── train/  cropped_images/  masks/  features.csv  concepts/
│   ├── val/    cropped_images/  masks/  features.csv  concepts/
│   └── test/   cropped_images/  masks/  features.csv  concepts/
├── explanations/
│   ├── gradcam/          # saliency maps, per test image
│   ├── gnn_explainer/    # keypoint/edge importance
│   ├── shap/             # per-feature attribution plots
│   ├── tcav/             # CAV scores per concept
│   └── cbm/              # concept-bottleneck predictions + accuracy comparison
├── evaluation/
│   ├── fidelity_stability_results.json
│   └── figures/
└── expert_study/
    ├── questionnaire.pdf
    └── session_notes/
```

---

## Notes

- This README will be updated as each planned script is implemented — "Planned" entries above become "Done" with a one-line purpose update once written, following the same convention as `fed_manual 1`'s `README.md`.
- Any script here that reads from `fed_manual 1/outputs/models/` treats those checkpoints as read-only inputs.
