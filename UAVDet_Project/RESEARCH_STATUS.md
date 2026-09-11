# Research Status Report
## Small Object Detection in UAV/Drone Imagery — SOA-FasterRCNN

**Last Updated:** 2026-08-21  
**Project:** Final Year Research Project  
**Dataset:** VisDrone-DET  
**Base Detector:** Faster R-CNN + ResNet-50-FPN (as instructed by mentor)

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Repository Audit | ✅ COMPLETE — see PROJECT_AUDIT.md |
| Phase 2 | Literature/Gap Analysis | ✅ COMPLETE — see docs/RESEARCH_GAP.md |
| Phase 3 | Architecture Proposal | ✅ COMPLETE — see docs/NOVELTY_CLAIM.md, docs/EXPERIMENT_PLAN.md |
| Phase 4 | Implementation Plan | ✅ COMPLETE — research/ directory structure created |
| Phase 5 | Baseline Implementation | ✅ COMPLETE — research/models/baseline_fasterrcnn.py |
| Phase 6 | Baseline Training | ⏳ PENDING — Requires VisDrone dataset + GPU |
| Phase 7 | Attention Experiment | ⏳ PENDING — Requires Phase 6 complete |
| Phase 8 | Second Module Experiment | ⏳ PENDING — Requires Phase 6 complete |
| Phase 9 | Combined Model | ⏳ PENDING — Requires Phases 7+8 complete |
| Phase 10 | Ablation Study | ⏳ PENDING — Requires Phases 6-9 complete |
| Phase 11 | Error Analysis | ⏳ PENDING — Requires Phase 10 complete |
| Phase 12 | Final Comparison | ⏳ PENDING — Requires Phase 11 complete |
| Phase 13 | Paper Preparation | ⏳ PENDING — see docs/PAPER_NOTES.md (template ready) |

---

## What Is Already Implemented

### Code (research/ directory)
- ✅ `research/configs/fasterrcnn_visdrone.yaml` — Full research config with debug/research modes
- ✅ `research/datasets/visdrone.py` — VisDrone COCO-format dataset loader with UAV-specific augmentation
- ✅ `research/models/baseline_fasterrcnn.py` — Baseline Faster R-CNN factory
- ✅ `research/models/attention.py` — SE, CBAM, ECA, Coordinate Attention, FPNLateralAttention
- ✅ `research/models/proposed_module.py` — Scale-Aware Context Module (SACM) for RPN
- ✅ `research/models/proposed_model.py` — Full proposed model assembly
- ✅ `research/training/trainer.py` — Full research trainer with AP_small logging
- ✅ `research/evaluation/metrics.py` — COCO metrics (mAP, AP_small/medium/large, PR curves)
- ✅ `research/evaluation/evaluate.py` — Full evaluation pipeline
- ✅ `research/evaluation/error_analysis.py` — Failure case analysis
- ✅ `research/visualization/visualize_predictions.py` — Qualitative results + training curves
- ✅ `research/run_experiment.py` — Unified experiment runner (4 experiments)

### Documentation (docs/ directory)
- ✅ `PROJECT_AUDIT.md` — Full repository audit report
- ✅ `docs/RESEARCH_GAP.md` — Literature review + gap analysis
- ✅ `docs/NOVELTY_CLAIM.md` — Honest novelty claim table
- ✅ `docs/EXPERIMENT_PLAN.md` — Ablation study design
- ✅ `docs/PAPER_NOTES.md` — Full paper draft template

### Existing (training/ directory — kept unchanged)
- ✅ `training/scripts/train_fasterrcnn.py` — Original training script (baseline reference)
- ✅ `training/configs/fasterrcnn.yaml` — Original config (partially outdated)
- ✅ `training/configs/uav.yaml` — VisDrone class names
- ✅ `training/hpc/run_interactive.sh` — HPC launcher script
- ✅ `training/README_HPC.md` — HPC setup instructions

---

## What Was Changed

1. **Created `research/` directory** with complete modular research codebase
2. **Created `docs/` directory** with all required research documentation
3. **Created `experiments/` directory structure** for training outputs
4. **Created `results/baseline/plots/` and `results/baseline/predictions/`** directories
5. **Did NOT modify** any existing code in `training/`, `backend/`, or `frontend/`

---

## What Remains (Pre-Training)

**Required before Phase 6:**
- [ ] Download VisDrone-DET dataset from https://github.com/VisDrone/VisDrone-Dataset
- [ ] Convert VisDrone .txt annotations to COCO JSON format using `training/README_HPC.md`
- [ ] Verify dataset structure: `training/datasets/uav/images/train/` + `annotations/`
- [ ] Upload to HPC at hpckongu.edu
- [ ] Install requirements: `pip install -r requirements.txt` (see requirements.txt)

**Required before Phase 13:**
- [ ] Run all 4 experiments
- [ ] Fill ABLATION_RESULTS.csv with actual results
- [ ] Complete ERROR_ANALYSIS.md with experimental findings
- [ ] Generate qualitative visualization images
- [ ] Fill [RESULT TO BE FILLED AFTER TRAINING] in PAPER_NOTES.md

---

## Current Baseline Status

**NOT YET TRAINED.**

All baseline metrics are NOT YET VERIFIED:
- mAP@0.5: NOT YET VERIFIED
- mAP@0.5:0.95: NOT YET VERIFIED
- AP_small: NOT YET VERIFIED
- AP_medium: NOT YET VERIFIED
- AP_large: NOT YET VERIFIED
- Precision: NOT YET VERIFIED
- Recall: NOT YET VERIFIED
- F1: NOT YET VERIFIED
- FPS: NOT YET VERIFIED
- Parameter count: ~46.3M (estimated from model architecture; NOT YET VERIFIED experimentally)
- GFLOPs: NOT YET VERIFIED

---

## Proposed Contributions

### Contribution 1: FPN Lateral Coordinate Attention
**Status:** POTENTIAL NOVELTY — REQUIRES VERIFICATION  
**Implementation:** Complete (research/models/attention.py)  
**Effect on metrics:** NOT YET VERIFIED — Expected to improve AP_small

### Contribution 2: Scale-Aware Context Module (SACM)
**Status:** POTENTIAL NOVELTY — REQUIRES VERIFICATION  
**Implementation:** Complete (research/models/proposed_module.py)  
**Effect on metrics:** NOT YET VERIFIED — Expected to improve Precision, reduce false positives

---

## Experiments Completed

| Experiment | Status | Results |
|-----------|--------|---------|
| E1: Baseline Faster R-CNN | ⏳ PENDING | NOT YET VERIFIED |
| E2: + Coordinate Attention | ⏳ PENDING | NOT YET VERIFIED |
| E3: + SACM | ⏳ PENDING | NOT YET VERIFIED |
| E4: Full SOA-FasterRCNN | ⏳ PENDING | NOT YET VERIFIED |

---

## Current Best Result

**NOT YET VERIFIED.**  
No training has been completed. The best result will be updated here after training.

---

## Comparison With Baseline

NOT YET VERIFIED. Comparison will be added after experiments complete.

---

## Whether Claimed Novelty is Supported

**Status:** POTENTIAL NOVELTY — REQUIRES VERIFICATION

The novelty claims have been carefully formulated (see docs/NOVELTY_CLAIM.md) to:
1. Not claim any individual component as novel
2. Claim the SPECIFIC COMBINATION as potentially novel
3. Require experimental validation to support the claim

No claim is made that exceeds what literature evidence supports.

---

## What Still Needs to Be Done Before Publication

1. **Dataset preparation** — Download and convert VisDrone annotations
2. **GPU training** — Run all 4 experiments (~64h total compute)
3. **Literature cross-check** — Verify no identical approach exists in papers not yet reviewed
4. **Fill all [RESULT TO BE FILLED] placeholders** in PAPER_NOTES.md
5. **Generate qualitative figures** — Prediction visualizations for paper
6. **Generate all plots** — Loss curves, mAP curves, PR curves, ablation charts
7. **Complete error analysis** — Fill ERROR_ANALYSIS.md with experimental findings
8. **Write final comparison** — Baseline vs proposed, with honest UAVDet comparison
9. **Review novelty claims** — Update based on experimental results
10. **Write paper** — Using PAPER_NOTES.md as scaffold
