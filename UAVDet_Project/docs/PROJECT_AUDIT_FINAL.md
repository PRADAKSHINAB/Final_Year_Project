# PROJECT AUDIT FINAL
## UAV Small-Object Detection — VisDrone + Faster R-CNN

---

## 1. Current Architecture
- **E1 (Baseline Faster R-CNN):** Standard Faster R-CNN + ResNet-50 + FPN backbone (41.35M parameters, 41.12M trainable). Anchors: 16, 32, 64, 128, 256 px; aspect ratios: 0.5, 1.0, 2.0, 3.0.
- **E2 (Coordinate Attention):** E1 + Coordinate Attention (Hou et al. 2021) applied at FPN output P2/P3 levels (41.36M parameters, 41.14M trainable).
- **E3 (Scale-Aware Context Module):** E1 + Scale-Aware Context Module (SACM) replacing the standard RPN head with multi-scale dilated convolutions and scale bin weighting (41.58M parameters, 41.36M trainable).
- **E4 (Full Proposed SOA-FasterRCNN):** E2 + E3 combined model incorporating both Coordinate Attention at FPN output P2/P3 and SACM in the RPN (41.60M parameters, 41.37M trainable).

---

## 2. Files Required for Research
- `research/models/baseline_fasterrcnn.py` — Baseline Faster R-CNN builder
- `research/models/attention.py` — Coordinate Attention implementation
- `research/models/proposed_module.py` — Scale-Aware Context Module (SACM)
- `research/models/proposed_model.py` — SOA-FasterRCNN model builder (E1–E4)
- `research/datasets/visdrone.py` — VisDrone COCO-format dataset loader (v2 transform fixed)
- `research/prepare_dataset.py` — VisDrone TXT → COCO JSON converter & loader validator
- `research/training/trainer.py` — Main training loop with SGD, Cosine Annealing, AMP
- `research/configs/fasterrcnn_visdrone.yaml` — Config file (split-aware dataset paths)
- `research/run_experiment.py` — Single experiment runner script
- `research/run_all_experiments.py` — Sequential runner script for E1–E4
- `research/check_hpc.py` — HPC H200 hardware & environment validator

---

## 3. Files Required for Inference
- `research/inference.py` — Single-image and multi-image directory inference script (`--model` / `--checkpoint` flags supported)
- Model checkpoints under `results/E{1..4}_*/checkpoints/best.pth`

---

## 4. Files Required for Evaluation
- `research/pipeline.py` — Complete result pipeline (predictions.json, COCO metrics, confusion matrix, small object metrics, efficiency benchmarks)
- `research/evaluate_all.py` — Cross-experiment evaluation, ablation table, publication figures, qualitative grid
- `research/evaluation/metrics.py` — pycocotools COCOeval wrapper & speed benchmarking

---

## 5. Files Required for Publication
All results generated under `results/`:
- `results/ablation_table.csv` — Primary quantitative results
- `results/ablation_map.png` — mAP bar chart
- `results/ablation_ap_small.png` — AP_small comparison figure
- `results/ablation_fps.png` — Throughput comparison
- `results/model_comparison.png` — Multi-metric grouped bar chart
- `results/qualitative_comparison/` — Original | GT | E1 | E2 | E3 | E4 side-by-side comparison images

---

## 6. Files That Can Safely Be Removed / Excluded
- `research/output_pipeline.py` (legacy superseded by `pipeline.py`)
- `frontend/` and `backend/` (web app directories, not needed for research execution)
- `training/` (legacy scripts, superseded by `research/`)
- `__pycache__` directories and temporary `.jpg`/`.json` test artifacts

---

## 7. Dataset Structure
- **Raw Input:** VisDrone2019-DET TXT format
  - Train images: `Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images/` (6,471 images)
  - Train annotations: `Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/annotations/` (6,471 TXT)
  - Val images: `Dataset/VisDrone2019-DET-val/images/` (500 images)
  - Val annotations: `Dataset/VisDrone2019-DET-val/annotations/` (500 TXT)
- **Converted Output (COCO JSON):**
  - Train JSON: `Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/annotations/instances_train.json` (343,197 annotations)
  - Val JSON: `Dataset/VisDrone2019-DET-val/annotations/instances_val.json` (40,395 annotations)

---

## 8. Configuration Structure
Configuration file: `research/configs/fasterrcnn_visdrone.yaml`
- `dataset.train_root`: Split-aware path to training root
- `dataset.val_root`: Split-aware path to validation root
- `debug`: 5 epochs, 640px resolution, 5% data subset
- `research`: 100 epochs, 1280px resolution, 100% full dataset

---

## 9. Dependency Requirements
`requirements.txt`:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `pycocotools>=2.0.6`
- `opencv-python>=4.7.0`
- `Pillow>=9.5.0`
- `matplotlib>=3.7.0`
- `numpy>=1.24.0`
- `PyYAML>=6.0`
- `fvcore>=0.1.5`

---

## 10. Known Bugs & Status
1. **BUG-1 (FIXED):** `visdrone.py` transform `TypeError` due to plain tensors passed to torchvision v2 `RandomIoUCrop`/`RandomZoomOut`. Fixed by wrapping image/boxes in `tv_tensors` using dict-based API in `_apply_transforms()`. Verified with 7/7 passing tests.
2. **BUG-2 (FIXED):** Config `dataset.root` mismatch with actual folder structure. Fixed by defining split-aware `train_root` and `val_root` in config and updating dataloader call.
3. **BUG-3 (FIXED):** Missing COCO JSON files. Resolved with `prepare_dataset.py`, converting all 6,471 train and 500 val TXT annotation files into valid COCO format (`instances_train.json` and `instances_val.json`).
4. **BUG-4 (DOCUMENTED):** Coordinate Attention in E2 is applied at FPN output (post-fusion) rather than lateral connections (pre-fusion). Retained as-is for ablation consistency.
5. **BUG-5 (DOCUMENTED):** Warmup scheduler resets when backbone unfreezes at epoch 5. Retained as-is across all 4 experiments for baseline fairness.

---

## 11. Fixes Performed
- Built `prepare_dataset.py` to auto-convert VisDrone TXT to COCO JSON and validate dataset loader integrity.
- Corrected torchvision v2 transform pipeline in `research/datasets/visdrone.py` to resolve `RandomIoUCrop` / `RandomZoomOut` tensor error.
- Updated `fasterrcnn_visdrone.yaml` with canonical VisDrone split paths.
- Built `check_hpc.py` to validate GPU environment and VRAM allocation.
- Built `check_dependencies.py` for automated import checking.
- Built `run_all_experiments.py` for automated sequential E1–E4 execution.
- Added `predictions.json` per-image detection schema compliance and exact publication figure file generation.
