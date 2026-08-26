# EXPERIMENT GUIDE
## SOA-FasterRCNN — UAV Small-Object Detection

---

## Prerequisites

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify pycocotools
python -c "from pycocotools.coco import COCO; print('OK')"

# Optional — for GFLOPs measurement
pip install fvcore
```

**Dataset path required:**
```
training/datasets/uav/
    images/
        train/   ← VisDrone training images (.jpg)
        val/     ← VisDrone validation images (.jpg)
    annotations/
        instances_train.json
        instances_val.json
```

---

## Step 0 — Sanity Test (no dataset required)

Verify the entire pipeline works before training:

```bash
python research/sanity_test.py --device cpu
```

Expected: `19 PASS  0 FAIL`
Generates real output files in `research/results/E1_baseline/` for inspection.

---

## Step 1 — Debug Run (ALWAYS do this first)

Uses 5% of training data, 5 epochs, 640px images. Takes ~10 minutes on GPU.

```bash
python research/run_experiment.py --experiment baseline      --mode debug --device cuda
python research/run_experiment.py --experiment attention     --mode debug --device cuda
python research/run_experiment.py --experiment second_module --mode debug --device cuda
python research/run_experiment.py --experiment final_model   --mode debug --device cuda
```

After each debug run verify these exist:
- `research/results/E1_baseline/checkpoints/best.pth`
- `research/results/E1_baseline/predictions/image_*.jpg`
- `research/results/E1_baseline/metrics/metrics.json`
- `research/results/E1_baseline/logs/training_log.csv`

---

## Step 2 — How to Train Each Experiment

### E1: Baseline Faster R-CNN
```bash
python research/run_experiment.py \
    --experiment baseline \
    --mode research \
    --device cuda \
    --num_vis 50
```
Architecture: Standard Faster R-CNN + ResNet-50 + FPN
Output: `research/results/E1_baseline/`

### E2: + Coordinate Attention
```bash
python research/run_experiment.py \
    --experiment attention \
    --mode research \
    --device cuda \
    --num_vis 50
```
Architecture: E1 + Coordinate Attention applied at FPN output P2/P3
Output: `research/results/E2_attention/`

### E3: + Scale-Aware Context Module
```bash
python research/run_experiment.py \
    --experiment second_module \
    --mode research \
    --device cuda \
    --num_vis 50
```
Architecture: E1 + ScaleAwareRPNHead (dilated context + scale bins)
Output: `research/results/E3_sacm/`

### E4: Full SOA-FasterRCNN (CA + SACM)
```bash
python research/run_experiment.py \
    --experiment final_model \
    --mode research \
    --device cuda \
    --num_vis 50
```
Architecture: E1 + Coordinate Attention + SACM
Output: `research/results/E4_full/`

---

## Step 3 — Where Checkpoints Are Stored

| File | Location | Description |
|------|----------|-------------|
| `best.pth` | `research/results/E1_baseline/checkpoints/best.pth` | Best validation mAP@50 |
| `last.pth` | `research/results/E1_baseline/checkpoints/last.pth` | Last epoch |

Best checkpoint is selected by **mAP@0.50 on the validation set**, measured every epoch.

To read checkpoint metadata:
```python
import torch
ckpt = torch.load("research/results/E1_baseline/checkpoints/best.pth")
print(ckpt["epoch"])                    # Epoch saved
print(ckpt["metrics"]["map50"])         # mAP@50 at that epoch
print(ckpt["model_name"])              # Model name
```

---

## Step 4 — Where Prediction JPGs Are Stored

```
research/results/E1_baseline/predictions/
    image_000001_prediction.jpg    ← overlay: boxes + class + confidence
    image_000002_prediction.jpg
    ...

research/results/E1_baseline/comparisons/
    image_000001_gt_vs_pred.jpg    ← LEFT=Ground Truth, RIGHT=Prediction
    image_000002_gt_vs_pred.jpg
    ...

research/results/qualitative_comparison/
    comparison_000001.jpg    ← Original|GT|E1|E2|E3|E4 side-by-side
    comparison_000002.jpg
    ...
```

All files are standard JPEG — open with any image viewer.

---

## Step 5 — Where Metrics Are Stored

```
research/results/E1_baseline/metrics/
    metrics.json             ← full COCO metrics
    per_class_metrics.csv    ← AP50, AP50_95 per class
    confusion_matrix.png     ← visual matrix
    confusion_matrix.csv     ← numerical matrix
    small_object_metrics.json
    efficiency.json

research/results/ablation_table.csv      ← E1 vs E2 vs E3 vs E4 table
research/results/final_comparison.csv   ← summary comparison
```

---

## Step 6 — Evaluation Only (after training)

Run the full output pipeline on an already-trained checkpoint:

```bash
python research/run_experiment.py \
    --experiment baseline \
    --eval-only \
    --device cuda
```

With a custom checkpoint path:
```bash
python research/run_experiment.py \
    --experiment baseline \
    --eval-only \
    --checkpoint path/to/custom.pth \
    --device cuda
```

---

## Step 7 — Compare All Four Experiments

After training all four, run the cross-experiment comparison:

```bash
python research/evaluate_all.py --device cuda --num_vis 20
```

This evaluates E1–E4 on the **exact same validation set** and generates:
- `research/results/ablation_table.csv`
- `research/results/final_comparison.csv`
- `research/results/comparison_plots/*.png`
- `research/results/qualitative_comparison/comparison_*.jpg`

---

## Step 8 — Single-Image Inference

```bash
python research/inference.py \
    --checkpoint research/results/E4_full/checkpoints/best.pth \
    --experiment final_model \
    --image path/to/test.jpg \
    --output path/to/result.jpg \
    --conf 0.30 \
    --device cuda
```

Prints detected objects and confidence scores. Saves result as JPEG.

---

## Step 9 — Multi-Image Batch Inference

```bash
python research/inference.py \
    --checkpoint research/results/E1_baseline/checkpoints/best.pth \
    --experiment baseline \
    --input_dir path/to/test_images/ \
    --output_dir research/results/E1_baseline/predictions/ \
    --conf 0.30 \
    --device cuda
```

Processes all `.jpg`, `.jpeg`, `.png` images in the directory.

---

## Step 10 — Generate Ablation Results for Paper

1. Train all four experiments (Steps 2)
2. Run `evaluate_all.py` (Step 7)
3. Use `research/results/ablation_table.csv` as Table 1/2 in the paper
4. Use `research/results/qualitative_comparison/` images for Figure in paper
5. Use `research/results/comparison_plots/AP_small_comparison.png` for AP_small figure

---

## Important: Do NOT Modify These Files

```
research/models/baseline_fasterrcnn.py    ← E1 model
research/models/attention.py              ← CA module
research/models/proposed_module.py        ← SACM module
research/models/proposed_model.py         ← E2/E3/E4 builders
research/training/trainer.py              ← training loop
research/datasets/visdrone.py             ← dataset loader
research/configs/fasterrcnn_visdrone.yaml ← shared config
```

---

## Training Configuration Reference

All four experiments use **identical** training settings from `research/configs/fasterrcnn_visdrone.yaml`:

| Parameter | Debug | Research |
|-----------|-------|---------|
| Image size | 640px | 1280px |
| Epochs | 5 | 100 |
| Batch size | 2 | 4 |
| Data fraction | 5% | 100% |
| Workers | 2 | 8 |
| Optimizer | SGD lr=0.005 | SGD lr=0.005 |
| LR Schedule | Cosine + 3-ep warmup | same |
| Backbone freeze | First 5 epochs | same |
| Anchors | 16,32,64,128,256 px | same |
| Seed | 42 | 42 |
