# Evaluation & Testing Guide — UAV Small-Object Detection

This document clarifies the evaluation methodology, metrics definitions, and distinction between ground-truth validation and custom image inference.

---

## 1. Ground-Truth Validation (VisDrone Validation Set)

When evaluating models on the VisDrone validation dataset (which has annotated bounding boxes and category labels), the pipeline computes official COCO benchmarks using `pycocotools.cocoeval.COCOeval`:

* **mAP@50:** Mean Average Precision at IoU threshold = 0.50
* **mAP@50:95:** Primary COCO benchmark (averaged across IoU thresholds from 0.50 to 0.95 with step 0.05)
* **AP_small:** Average Precision for small objects ($\text{area} < 32^2 \text{ px}^2 = 1024 \text{ px}^2$)
* **AP_medium:** Average Precision for medium objects ($32^2 \le \text{area} < 96^2 \text{ px}^2$)
* **AP_large:** Average Precision for large objects ($\text{area} \ge 96^2 \text{ px}^2$)
* **Precision / Recall / F1:** Calculated at the optimal F1 threshold from the precision-recall curve
* **Per-Class Metrics:** Individual AP50, AP50:95, Precision, Recall, and F1 for all 10 VisDrone categories:
  1. `pedestrian`
  2. `people`
  3. `bicycle`
  4. `car`
  5. `van`
  6. `truck`
  7. `tricycle`
  8. `awning-tricycle`
  9. `bus`
  10. `motor`
* **Confusion Matrix:** $11 \times 11$ matrix matching predictions to ground-truth objects using IoU $\ge 0.50$. Background false positives and false negatives are explicitly tracked.

---

## 2. Testing Custom / New UAV Images (No Ground Truth)

When testing your own captured UAV drone photos (which do **not** have ground truth annotations), the system performs detection inference without fabricating fake metrics:

* Draws bounding boxes with class-specific color coding
* Displays the predicted **VisDrone class name**
* Displays the predicted **confidence score** ($0.00 \dots 1.00$)
* Measures and displays **inference latency** (milliseconds) and **device** (GPU/CPU)
* Saves clean output images into `results/predictions/<image_name>_detected.jpg`

---

## 3. Evaluation Commands

### Re-evaluate an existing checkpoint without re-training:
```bash
python research/run_experiment.py \
    --experiment final_model \
    --eval-only \
    --checkpoint results/E4_full/checkpoints/best.pth \
    --device cuda
```

### Generate cross-experiment comparative tables and plots:
```bash
python research/evaluate_all.py
```
