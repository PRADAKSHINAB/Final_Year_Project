# Training Guide — Small-Object-Aware Faster R-CNN (SOA-FasterRCNN)

## Overview of Research Experiments

| Exp ID | Experiment Name | Key Technical Component | Primary Output Directory |
| :--- | :--- | :--- | :--- |
| **E1** | Baseline Faster R-CNN | ResNet-50 + FPN + 16px anchor tuning | `results/E1_baseline/` |
| **E2** | Coordinate Attention | E1 + Coordinate Attention at FPN Lateral Connections | `results/E2_attention/` |
| **E3** | Scale-Aware Context (SACM) | E1 + Scale-Aware RPN Head with Multi-Bin Context | `results/E3_sacm/` |
| **E4** | Full Proposed SOA-FasterRCNN | Combined E2 (Attention) + E3 (SACM RPN) | `results/E4_full/` |

---

## Output Directory Structure per Experiment

For every experiment trained, the system automatically creates:

```text
results/E4_full/
├── checkpoints/
│   ├── best.pth               # Model with highest validation mAP@50
│   └── last.pth               # Checkpoint at final epoch
├── predictions/
│   ├── image_000001.jpg       # Annotated validation predictions
│   └── predictions.json       # Structured detection records
├── comparisons/
│   └── image_000001_gt_vs_pred.jpg # Side-by-side: Ground Truth vs Prediction
├── metrics/
│   ├── metrics.json           # mAP50, mAP50_95, AP_s/m/l, Prec, Rec, F1, FPS, GFLOPs
│   ├── per_class_metrics.csv  # AP50, AP50_95, precision, recall for all 10 classes
│   ├── confusion_matrix.csv   # 11x11 detection confusion matrix
│   ├── confusion_matrix.png   # Log-scaled confusion matrix heatmap
│   ├── small_object_metrics.json # Dedicated sub-32px object detection statistics
│   └── efficiency.json        # Parameters, GFLOPs, latency, GPU memory
├── plots/
│   ├── train_loss.png         # Total loss, RPN loss, ROI loss curves
│   ├── val_loss.png           # Validation mAP trajectory
│   ├── map_curve.png          # mAP@50 progression curve
│   ├── learning_rate.png      # Cosine annealing warmup LR curve
│   ├── ap_size_curve.png      # AP_small vs AP_medium vs AP_large
│   └── small_object_analysis.png # Pie chart & TP/FP/FN bar chart for small objects
└── logs/
    ├── training_log.csv       # Per-epoch loss and metric history
    ├── experiment_config.yaml # Exact hyperparameters used
    └── experiment_summary.txt # Complete reproducibility summary
```

---

## Configuration & Hyperparameters

Configured in [`research/configs/fasterrcnn_visdrone.yaml`](file:///c:/Users/prada/Downloads/Final%20Year%20Project/uavdet-system/research/configs/fasterrcnn_visdrone.yaml):

* **Image Size:** 1280 (Research mode) / 640 (Debug mode)
* **Optimizer:** SGD (`lr=0.005`, `momentum=0.9`, `weight_decay=0.0001`)
* **LR Scheduler:** Cosine Annealing with 3 epochs linear warmup
* **Backbone Freeze:** Initial 5 epochs to stabilize RPN heads
* **Anchor Scales:** `(16, 32, 64, 128, 256)` pixels
* **Aspect Ratios:** `(0.5, 1.0, 2.0, 3.0)`
* **Mixed Precision (AMP):** Enabled automatically on CUDA
