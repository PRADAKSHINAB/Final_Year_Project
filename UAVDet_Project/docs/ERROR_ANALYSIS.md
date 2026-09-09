# Error Analysis Guide — VisDrone Small-Object Detection

## Overview
This document provides a systematic framework for analyzing detection failures in small-object aerial imagery using VisDrone-DET and Faster R-CNN.

---

## Failure Categories in UAV Small-Object Detection

### 1. Missed Small Objects (False Negatives)
- **Description:** Objects smaller than 32×32 pixels (or 16×16 pixels) missed by the RPN or ROI Head.
- **Root Cause:** Feature attenuation in deep FPN layers, low anchor IoU overlap with tiny ground-truth boxes, or aggressive NMS suppression.
- **Expected Module Impact:**
  - **SACM (E3/E4):** Multi-scale dilated context in RPN head increases recall on sub-32px objects.
  - **Coordinate Attention (E2/E4):** Calibrates spatial feature maps to preserve high-frequency positional detail at P2/P3.

### 2. False Positives (Background False Alarms)
- **Description:** Background noise (shadows, roof textures, road markings) misclassified as pedestrians or vehicles.
- **Root Cause:** High visual similarity between small background textures and distant pedestrians in low-resolution drone images.
- **Expected Module Impact:** SACM scale-bin classification filters out background proposals that lack scale context.

### 3. Localization Errors (Poor Bounding Box Alignment)
- **Description:** Detected object class is correct, but IoU with ground truth is < 0.50.
- **Root Cause:** Coarse feature grid stride at deeper FPN levels causing spatial jitter for small boxes.

### 4. Dense Scene Crowding & Occlusion
- **Description:** Overlapping pedestrians or parking-lot vehicles merged into single detection boxes or suppressed by NMS.
- **Root Cause:** Standard Greedy NMS with fixed IoU threshold (0.50) suppressing adjacent legitimate proposals.

### 5. Class Confusions
- **Description:** Pedestrians misclassified as People, or Cars misclassified as Vans/Trucks.
- **Root Cause:** High fine-grained visual similarity between categories at distance.

---

## How to Perform Error Analysis After Training

1. Run `python research/evaluate_all.py` to generate predictions and confusion matrices.
2. Inspect `results/E4_full/metrics/confusion_matrix.png` to identify top off-diagonal confusions.
3. Review `results/E4_full/metrics/small_object_metrics.json` for small-object TP/FP/FN breakdown.
4. Inspect `results/qualitative_comparison/` images to visually verify where E4 recovers objects missed by E1.
