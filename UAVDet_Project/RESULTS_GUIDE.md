# RESULTS GUIDE
## SOA-FasterRCNN — Understanding and Using Results

---

## Directory Structure

```
research/results/
├── E1_baseline/
│   ├── checkpoints/
│   │   ├── best.pth        ← Best model (highest mAP@50 on val)
│   │   └── last.pth        ← Last epoch model
│   ├── predictions/
│   │   ├── image_000001_prediction.jpg   ← JPEG: boxes + class + confidence
│   │   ├── image_000002_prediction.jpg
│   │   └── predictions.json             ← Machine-readable: x1,y1,x2,y2,confidence,class
│   ├── comparisons/
│   │   ├── image_000001_gt_vs_pred.jpg  ← LEFT=GT, RIGHT=Prediction
│   │   └── image_000002_gt_vs_pred.jpg
│   ├── metrics/
│   │   ├── metrics.json              ← Full COCO evaluation metrics
│   │   ├── per_class_metrics.csv     ← AP50/AP50_95 for all 10 classes
│   │   ├── confusion_matrix.png      ← Visual 10×10 matrix
│   │   ├── confusion_matrix.csv      ← Numerical 11×11 matrix (incl. background)
│   │   ├── small_object_metrics.json ← Small-object specific analysis
│   │   └── efficiency.json           ← Params, GFLOPs, FPS, GPU memory
│   ├── plots/
│   │   ├── train_loss.png        ← Training loss components per epoch
│   │   ├── val_loss.png          ← Validation mAP@50 + mAP@50:95 per epoch
│   │   ├── learning_rate.png     ← LR schedule curve
│   │   ├── map50_curve.png       ← mAP@50 over training
│   │   ├── map50_95_curve.png    ← mAP@0.5:0.95 over training
│   │   ├── ap_size_curve.png     ← AP_small / AP_medium / AP_large per epoch
│   │   └── small_object_analysis.png  ← Size distribution + TP/FP/FN chart
│   └── logs/
│       ├── training_log.csv      ← Per-epoch: loss, mAP, LR (from actual training)
│       ├── experiment_config.yaml
│       └── experiment_summary.txt
│
├── E2_attention/    (same structure)
├── E3_sacm/         (same structure)
├── E4_full/         (same structure)
│
├── qualitative_comparison/
│   ├── comparison_000001.jpg  ← Original | GT | E1 | E2 | E3 | E4
│   └── comparison_000002.jpg
│
├── comparison_plots/
│   ├── mAP_comparison.png
│   ├── AP_small_comparison.png
│   ├── full_comparison.png
│   ├── FPS_comparison.png
│   └── small_object_comparison.png
│
├── ablation_table.csv      ← Main ablation table for paper
└── final_comparison.csv    ← Summary with status column
```

---

## What Each Metric Means

### Primary Metrics (use in paper Table 1)

| Metric | Definition | Threshold |
|--------|-----------|-----------|
| **mAP@0.50** | Mean AP across all classes at IoU=0.5 | — |
| **mAP@0.5:0.95** | COCO standard: mean over IoU 0.5, 0.55, ..., 0.95 | — |
| **AP_small** | AP for objects with area < 1024 px² (< 32×32) | **Primary metric** |
| **AP_medium** | AP for objects 1024–9216 px² (32×32 – 96×96) | — |
| **AP_large** | AP for objects > 9216 px² (> 96×96) | — |
| **Precision** | True Positives / (True Positives + False Positives) at IoU=0.5 |
| **Recall** | True Positives / (True Positives + False Negatives) at IoU=0.5 |
| **F1** | Harmonic mean of Precision and Recall | — |

### How to Interpret AP_small

AP_small is the **most important metric** for this research project because:
- VisDrone images contain mostly small objects (pedestrians, vehicles from altitude)
- COCO defines "small" as objects with area < 32×32 pixels
- If AP_small for E2/E3/E4 > AP_small for E1, the proposed modules help small object detection

**Higher is better. Range: 0.0 to 1.0**

AP_small = 0.15 means the model detects 15% of small objects at the right location and class.
On VisDrone, even 0.10–0.20 is a reasonable result for a trained Faster R-CNN.

### Efficiency Metrics

| Metric | Definition |
|--------|-----------|
| **Parameters** | Total model weights (millions) |
| **GFLOPs** | Floating-point operations per forward pass |
| **Inference time** | Mean ms per image (100 warmup + measurements) |
| **FPS** | 1000 / mean_ms |
| **GPU memory** | Peak VRAM allocated during one forward pass |

---

## How to Compare E1/E2/E3/E4

### Correct interpretation

| Comparison | Purpose |
|-----------|---------|
| E2 vs E1 | Does Coordinate Attention improve detection? |
| E3 vs E1 | Does SACM improve small-object detection? |
| E4 vs E1 | Does the full proposed model improve over baseline? |
| E4 vs E2 | What does SACM add on top of CA? |
| E4 vs E3 | What does CA add on top of SACM? |

### From ablation_table.csv

Read `research/results/ablation_table.csv`. Each row is one experiment.
Compare `AP_small` column to assess the research contribution.

**Do NOT claim improvement unless AP_small(E4) > AP_small(E1).**
If the numbers are close (within measurement noise), acknowledge this honestly.

### Example ablation table format (values will be filled after training)
```
Experiment | Parameters | mAP50  | mAP50_95 | AP_small | AP_medium | AP_large | FPS
E1         |   41.35M   |  N/A   |    N/A   |    N/A   |    N/A    |    N/A   | N/A
E2 +CA     |   41.36M   |  N/A   |    N/A   |    N/A   |    N/A    |    N/A   | N/A
E3 +SACM   |   41.58M   |  N/A   |    N/A   |    N/A   |    N/A    |    N/A   | N/A
E4 Full    |   41.60M   |  N/A   |    N/A   |    N/A   |    N/A    |    N/A   | N/A
```

---

## Which Plots to Use in the Paper

### Table 1: Main Quantitative Results
**File:** `research/results/ablation_table.csv`
Copy mAP50, mAP50_95, AP_small, Parameters, FPS columns.

### Table 2: Ablation Study
Same file, focus on AP_small improvement row-by-row.

### Figure: mAP Comparison Bar Chart
**File:** `research/results/comparison_plots/full_comparison.png`
Shows all metrics side-by-side for E1-E4.

### Figure: AP_small Comparison
**File:** `research/results/comparison_plots/AP_small_comparison.png`
Single-metric bar chart for the primary research claim.

### Figure: Qualitative Detection Examples
**File:** `research/results/qualitative_comparison/comparison_*.jpg`
Shows Original | GT | E1 | E2 | E3 | E4 on the same image.
Pick images that clearly show small objects being detected or missed.

### Figure: Training Loss Curve
**File:** `research/results/E1_baseline/plots/train_loss.png`

### Figure: mAP Training Curve
**File:** `research/results/E1_baseline/plots/val_loss.png`
(Contains mAP@0.50 and mAP@0.5:0.95 over epochs)

### Figure: Confusion Matrix
**File:** `research/results/E1_baseline/metrics/confusion_matrix.png`

### Figure: Small-Object Analysis
**File:** `research/results/E1_baseline/plots/small_object_analysis.png`

---

## How the Confusion Matrix Works

The confusion matrix is a **detection confusion matrix**, not a classification matrix.

- **Rows** = Ground-Truth class
- **Columns** = Predicted class
- **IoU threshold** = 0.50 (COCO standard)
- **Index 0** = background / no-match

Matching rule:
1. For each prediction (confidence ≥ 0.25), find the GT box with highest IoU ≥ 0.5 in the same image.
2. If matched: `matrix[gt_class, pred_class] += 1`
3. If no GT match (false positive): `matrix[0, pred_class] += 1`
4. Unmatched GT (false negative / missed): `matrix[gt_class, 0] += 1`

**Diagonal = correct detections.**
**Off-diagonal = class confusions** (e.g., pedestrian predicted as people).
**Column 0 = false positives.**
**Row 0 = missed objects.**

---

## predictions.json Format

Each detection entry in `predictions/predictions.json`:
```json
{
  "image_id": 1,
  "category_id": 4,
  "category_name": "car",
  "x1": 120.5,
  "y1": 88.2,
  "x2": 210.3,
  "y2": 156.7,
  "width": 89.8,
  "height": 68.5,
  "confidence": 0.923456
}
```

All values come directly from model output. No values are fabricated.

---

## Important Notes

1. **All metrics are from pycocotools COCOeval** — the same library used in the COCO challenge and all major detection papers.

2. **AP = -1.000** in COCOeval output means no ground-truth boxes exist for that size category in the evaluated images. This is normal and expected.

3. **N/A in CSVs** means the experiment has not yet been trained. Run training first.

4. **Never fabricate results.** If a metric shows a small or negative improvement, report it honestly. The research contribution is the method, not inflated numbers.

5. **Confidence threshold for visualisations** is `conf=0.25` by default. This is lower than the paper-standard `0.5` to show more detections. COCO evaluation uses all predictions regardless of threshold.
