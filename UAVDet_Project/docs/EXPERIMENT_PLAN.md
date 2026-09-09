# Experiment Plan
## Ablation Study Design for SOA-FasterRCNN

---

## Overview

Four experiments are planned to isolate the contribution of each proposed component.
This ablation structure is required for peer-reviewed publication.

**Base framework:** Faster R-CNN + ResNet-50-FPN  
**Dataset:** VisDrone-DET (full training + validation splits)  
**Seed:** 42 (single seed; multiple seeds not feasible due to compute constraints — documented as limitation)

---

## Experiment Definitions

### Experiment 1: Baseline Faster R-CNN
**Purpose:** Establish the research baseline for all comparisons.

| Parameter | Value |
|-----------|-------|
| Architecture | Faster R-CNN + ResNet50-FPN V2 |
| Backbone pretrain | ImageNet-1K V2 |
| Attention | NONE |
| SACM | NONE |
| Anchor sizes | 16, 32, 64, 128, 256 px |
| Aspect ratios | 0.5, 1.0, 2.0, 3.0 |
| Image size | 1280 x 1280 |
| Epochs | 100 |
| Batch size | 4 |
| Optimizer | SGD (lr=0.005, momentum=0.9, wd=0.0001) |
| LR schedule | Cosine annealing + 3-epoch warmup |
| Backbone freeze | First 5 epochs |
| AMP | Yes |
| Seed | 42 |

**Command:**
```bash
python research/run_experiment.py --experiment baseline --mode research
```

**Expected output:** Establishes baseline mAP@50, mAP@0.5:0.95, AP_small

---

### Experiment 2: Baseline + Coordinate Attention
**Purpose:** Isolate contribution of the attention module (Contribution 1).

| Parameter | Value |
|-----------|-------|
| Base | Same as Experiment 1 |
| Addition | Coordinate Attention at FPN lateral connections (P2 + P3) |
| Attention type | Coordinate Attention (Hou et al., 2021) |
| Reduction ratio | 32 |
| Location | FPN lateral connections (P2 and P3 levels) |
| SACM | NONE |

**Command:**
```bash
python research/run_experiment.py --experiment attention --mode research
```

**Key metric to watch:** AP_small (primary claim)
**Expected effect:** Improved AP_small; minimal effect on AP_large

---

### Experiment 3: Baseline + SACM Only
**Purpose:** Isolate contribution of Scale-Aware Context Module (Contribution 2).

| Parameter | Value |
|-----------|-------|
| Base | Same as Experiment 1 |
| Addition | ScaleAwareRPNHead replacing standard RPNHead |
| Scale bins | 4 |
| Context channels | 64 |
| Attention | NONE |

**Command:**
```bash
python research/run_experiment.py --experiment second_module --mode research
```

**Key metric to watch:** Precision, false positive reduction, AP_small
**Expected effect:** Reduced false positives; improved precision in dense scenes

---

### Experiment 4: Full Proposed Model (SOA-FasterRCNN)
**Purpose:** Evaluate combined effect; should exceed Experiments 2 + 3 individually.

| Parameter | Value |
|-----------|-------|
| Base | Same as Experiment 1 |
| Contribution 1 | Coordinate Attention at FPN lateral (P2+P3) |
| Contribution 2 | Scale-Aware Context Module in RPN |
| Model name | SOA-FasterRCNN (Small-Object-Aware Faster R-CNN) |

**Command:**
```bash
python research/run_experiment.py --experiment final_model --mode research
```

**Key metric to watch:** mAP@0.5, mAP@0.5:0.95, AP_small (all should improve vs baseline)

---

## Ablation Results Table (Template)

Results to be filled after training. Current values are placeholders.

| Model | mAP50 | mAP50-95 | AP_small | AP_medium | AP_large | Precision | Recall | F1 | FPS | Params(M) | GFLOPs |
|-------|-------|----------|----------|-----------|----------|-----------|--------|-----|-----|-----------|--------|
| E1: Baseline | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] |
| E2: +CA | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] |
| E3: +SACM | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] |
| E4: +CA+SACM | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] | [TBF] |

[TBF] = To Be Filled After Training — NEVER FABRICATED

---

## Debug Mode Verification (Run Before Full Training)

Before committing compute to 100-epoch runs, verify the pipeline works:

```bash
python research/run_experiment.py --experiment baseline --mode debug
```

Debug mode: 5% data, 5 epochs, 640px images, 2 workers.
Expected time: ~10-30 minutes on a GPU.

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|---------|---------|-------------|
| GPU VRAM | 8 GB (batch=2, 640px) | 16+ GB (batch=4, 1280px) |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB (dataset + results) | 100 GB |
| GPU time per experiment | ~8-16h (A100) | ~24h (lower GPU) |

**Total compute estimate:** 4 experiments x ~16h = ~64h GPU time
This is a significant but standard requirement for VisDrone-scale research.

---

## Statistical Reliability

Due to computational constraints, each experiment is run with ONE random seed (42).
This is a known limitation and must be stated in the paper:

*"Due to GPU resource constraints, ablation experiments are conducted with a single
random seed (42). We acknowledge that multiple seeds would improve statistical
reliability, and leave multi-seed validation as future work."*

---

## Comparison with UAVDet

The existing project contains UAVDet results. Comparison will use:

| Method | Training Setup | Dataset | Comparability |
|--------|---------------|---------|---------------|
| UAVDet (original paper) | Reported in paper | May differ | "Reported in original paper" — NOT directly comparable |
| Our Baseline | Our setup, VisDrone-full | VisDrone-DET | Direct comparison |
| Our E4 (SOA-FRCNN) | Our setup, VisDrone-full | VisDrone-DET | Direct comparison |

**Important:** We will NOT claim superiority over UAVDet unless trained under identical conditions.
