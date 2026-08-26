# PROJECT AUDIT REPORT
## UAVDet System — Full Repository Audit

**Date:** 2026-08-21  
**Auditor:** Research Engineer  
**Purpose:** Phase 1 — Systematic audit before any changes are made

---

## Repository Structure (As Found)

```
uavdet-system/
├── backend/                          # FastAPI web application
│   ├── .env.example                  # Environment variable template
│   ├── main.py                       # FastAPI app (title: UAVDet System API)
│   ├── models.py                     # SQLAlchemy ORM models
│   ├── schemas.py                    # Pydantic schemas
│   ├── requirements.txt              # Backend dependencies
│   ├── uavdet.db                     # SQLite production DB (committed — should be gitignored)
│   ├── test_uavdet.db                # SQLite test DB (committed — should be gitignored)
│   ├── api/routes/                   # Route handlers (admin, auth, detect, history, model, training)
│   ├── model_adapter/
│   │   └── fasterrcnn_adapter.py     # Inference bridge: webapp ↔ Faster R-CNN model
│   ├── services/
│   │   ├── auth.py                   # JWT authentication
│   │   ├── database.py               # DB connection + seeding
│   │   ├── detector.py               # Core detection orchestration (422 lines)
│   │   └── utils.py                  # File/IO utilities
│   └── tests/test_api.py             # Integration tests
│
├── frontend/                         # React/TypeScript web UI (Vite + Tailwind)
│   └── src/                          # UI source code
│
└── training/                         # ML training pipeline
    ├── README_HPC.md                 # HPC deployment guide (361 lines)
    ├── requirements.txt              # Training dependencies (INCLUDES unused ultralytics)
    ├── configs/
    │   ├── fasterrcnn.yaml           # Hyperparams (CONTAINS YOLO-SPECIFIC KEYS — ISSUE)
    │   └── uav.yaml                  # Dataset path + VisDrone 10-class config
    ├── datasets/uav/                 # EMPTY — dataset not included in repo
    ├── hpc/
    │   ├── run_interactive.sh        # JupyterHub launcher
    │   ├── train_slurm.sh            # SLURM job launcher
    │   └── package_weights.sh        # Weight packaging script
    └── scripts/
        └── train_fasterrcnn.py       # Main training script (588 lines)
```

---

## File Classification

### REQUIRED FILES — Core to the Project

| File | Role | Component |
|------|------|-----------|
| `training/scripts/train_fasterrcnn.py` | Main Faster R-CNN training + COCO evaluation | Training |
| `training/configs/uav.yaml` | Dataset path + VisDrone class mapping | Config |
| `training/hpc/run_interactive.sh` | JupyterHub launcher for HPC training | HPC |
| `training/requirements.txt` | PyTorch + torchvision + pycocotools | Dependencies |
| `backend/model_adapter/fasterrcnn_adapter.py` | Inference adapter: webapp → model | Inference |
| `backend/services/detector.py` | Detection orchestration service | Backend |
| `backend/services/database.py` | DB init + seeding | Backend |
| `backend/main.py` | FastAPI app factory | Backend |
| `backend/models.py` | SQLAlchemy ORM | Backend |
| `frontend/src/` | React web UI | Frontend |

---

### OPTIONAL FILES — Useful But Not Critical

| File | Role | Recommendation |
|------|------|----------------|
| `training/hpc/train_slurm.sh` | SLURM launcher (if cluster uses SLURM) | Keep |
| `training/hpc/package_weights.sh` | Packages output weights | Keep |
| `training/README_HPC.md` | HPC setup instructions | Keep |
| `backend/tests/test_api.py` | API integration tests | Keep |

---

### DUPLICATE / MISMATCHED FILES

| File | Issue | Action Required |
|------|-------|-----------------|
| `training/configs/fasterrcnn.yaml` | Contains YOLO-specific keys: `mosaic`, `mixup`, `copy_paste`, `dfl`, `close_mosaic` — NOT used by train_fasterrcnn.py | Rewrite with proper Faster R-CNN keys AFTER CONFIRMATION |

---

### OBSOLETE / UAVDet-LEGACY FILES

| File / Line | Issue | Action |
|-------------|-------|--------|
| `training/requirements.txt` line 3: `ultralytics>=8.1.34` | YOLO framework; unused in Faster R-CNN training | Remove AFTER CONFIRMATION |
| `backend/services/detector.py` env var defaults | `UAVDET_MODEL_NAME="UAVDet"` branding | Acceptable; update to reflect new model name |
| `backend/uavdet.db` | Production SQLite DB committed to repo | Should be gitignored |
| `backend/test_uavdet.db` | Test SQLite DB committed to repo | Should be gitignored |

---

### EXPERIMENTAL / EMPTY DIRECTORIES

| Directory | Status | Action |
|-----------|--------|--------|
| `training/datasets/uav/` | EMPTY — no dataset | Populate with VisDrone-DET data |
| `training/weights/` | EMPTY — no trained models | Populated after training |
| `backend/weights/` | EMPTY — no inference weights | Requires best.pt from training |

---

### FILES TO REMOVE ONLY AFTER CONFIRMATION

> ⚠️ DO NOT DELETE without explicit user confirmation.

| File | Reason for Potential Removal |
|------|------------------------------|
| `ultralytics` in `training/requirements.txt` | Legacy YOLO dependency not needed |
| YOLO-specific keys in `training/configs/fasterrcnn.yaml` | Misleading to readers; not consumed by code |
| `backend/uavdet.db` | Should be gitignored |
| `backend/test_uavdet.db` | Should be gitignored |

---

## Existing Code Quality Assessment

### train_fasterrcnn.py — FUNCTIONAL BASELINE
✅ Small-object anchor scales (16, 32, 64, 128, 256 px)  
✅ Extended aspect ratios (0.5, 1.0, 2.0, 3.0)  
✅ Custom RPN top-K settings for dense scenes  
✅ Backbone freeze/unfreeze schedule  
✅ COCO mAP evaluation via pycocotools  
✅ Mixed precision training (AMP)  
✅ CSV logging  
✅ Best model saving  

### Gaps in Existing Code (for research purposes)
❌ AP_small / AP_medium / AP_large not reported separately  
❌ No per-class AP  
❌ No PR curve generation  
❌ No debug mode (always full config)  
❌ No attention module integration  
❌ No second research module  
❌ Evaluation builds GT from runtime (less accurate than loading from annotation file directly)  

These gaps are addressed by the new `research/` directory.

---

## Files Related to Prior UAVDet Implementation

Evidence of prior YOLO-based UAVDet system:
1. `training/requirements.txt` — `ultralytics>=8.1.34`
2. `training/configs/fasterrcnn.yaml` — YOLO augmentation keys (mosaic, mixup, copy_paste, dfl)
3. `backend/main.py` — FastAPI titled "UAVDet System API"
4. `training/README_HPC.md` — References "YOLO baseline" in comparison section

The training script was retrofitted from YOLO to Faster R-CNN, but config and requirements
still contain YOLO artifacts.

---

## New Research Structure Created

```
research/                            # Research code (NEW)
│   ├── configs/fasterrcnn_visdrone.yaml   # Debug + research modes
│   ├── datasets/visdrone.py               # Clean VisDrone loader
│   ├── models/                            # Baseline + attention + proposed
│   ├── training/trainer.py                # Research trainer with AP_small
│   ├── evaluation/                        # Full COCO eval + error analysis
│   ├── visualization/                     # Qualitative results
│   └── run_experiment.py                  # Unified runner
docs/                                # Documentation (NEW)
│   ├── RESEARCH_GAP.md
│   ├── NOVELTY_CLAIM.md
│   ├── EXPERIMENT_PLAN.md
│   └── PAPER_NOTES.md
PROJECT_AUDIT.md                     # This file (NEW)
RESEARCH_STATUS.md                   # Status tracker (NEW)
ABLATION_RESULTS.csv                 # Results template (NEW)
ERROR_ANALYSIS.md                    # Error analysis template (NEW)
requirements.txt                     # Research requirements (NEW)
README.md                            # Project README (NEW)
```

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Required (keep unchanged) | 10+ files | No changes |
| Optional (keep) | 4 files | No changes |
| Duplicate/mismatched | 1 file | Rewrite after confirmation |
| Obsolete (YOLO legacy) | 2 items | Remove after confirmation |
| Empty directories | 3 | Populate with data + weights |
| New files created | 20+ | All in research/ and docs/ |

**Key finding:** The project is a full-stack webapp with a Faster R-CNN training script.
The existing training code is a solid functional baseline but lacks research-grade components
(attention modules, AP_small reporting, ablation support). New code is in `research/` only.
No existing code was modified or deleted.
