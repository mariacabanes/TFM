
## 📁 `fed_manual` — Dental Implant Analysis Pipeline

`fed_manual` contains the end-to-end pipeline used to go from raw CVAT annotation exports to trained models. Each script is numbered in the order it should run, and `run_all.py` can execute the whole pipeline automatically, skipping steps that have already completed.

| Step | Script | Purpose |
|---|---|---|
| 0 | `(0) run_all.py` | Master pipeline runner — executes steps 1–6 in order, auto-detecting already-completed steps and prompting before retraining. |
| 1 | `(1) prepare_dataset.py` | Extracts and merges multiple CVAT COCO 1.0 export zips into a single unified dataset (`annotations.json` + flat `images/` folder), remapping category/image/annotation IDs and resolving filename collisions. |
| 2 | `(2) dataset_analysis.py` | Computes dataset statistics, flags class imbalance, cleans annotations, and calculates per-category augmentation factors. Produces `analysis_results.json`. |
| 3 | `(3) split.py` | Splits the dataset into **train / val / test (70 / 15 / 15)**, copies images into the corresponding folders, and writes per-split COCO annotation files, along with distribution statistics. |
| 4 | `(4) keypoints_extraction.py` | Uses classic image-processing (contour detection, peak/valley detection on thread profiles, linear regression alignment) to classify each implant as **"Tipo Recto" (straight)** or **"Tipo U"**, extract anatomical keypoints (peaks, valleys, drops, top/bottom corners), and export them to `features.csv` along with cropped/annotated images. |
| 5 | `(5) augmentation.py` | Two-stage augmentation: (1) keypoint-aware **geometric** augmentation of crops (flips, rotations, translation, compound transforms for specific brands), and (2) **intensity-only** augmentation of raw images (brightness, gamma, Gaussian noise, CLAHE) — without altering geometry. Outputs `features_augmented.csv`. |
| 6 | `(6) train.py` | Trains the two core models: a **UNet** (segmentation) and an **ImplantClassifier** (diameter/model classification), with early stopping, checkpointing of the best model by validation IoU/accuracy, and model export. |

### Typical usage

```bash
# Run the entire pipeline (skips steps already completed)
python "(0) run_all.py"

# Run from a specific step onward
python "(0) run_all.py" --from 3

# Run a single step only
python "(0) run_all.py" --only 4

# Force re-run everything, including completed steps
python "(0) run_all.py" --force

# Re-run training without the interactive prompt
python "(0) run_all.py" --retrain
```

### Pipeline output structure

```
outputs/
├── dataset/                     # merged images + annotations.json (step 1)
├── analysis_results.json        # dataset stats + aug factors (step 2)
├── dataset_split/
│   ├── train/  images/  annotations.json  features.csv  features_augmented.csv
│   ├── val/    images/  annotations.json  features.csv
│   └── test/   images/  annotations.json
└── models/                      # trained UNet + ImplantClassifier checkpoints (step 6)
```

---

## Relationship between the two TFMs and the pipeline

The `fed_manual` pipeline is the shared data foundation for both projects: it prepares the periapical radiograph dataset, extracts the anatomical keypoints/graph features, and trains the base multimodal models (UNet for segmentation, EfficientNet/ResNet + GNN for classification) described in Paula Gisbert's thesis. This author's TFM builds on top of those trained models to add the post-hoc and concept-based explainability layer described above.
