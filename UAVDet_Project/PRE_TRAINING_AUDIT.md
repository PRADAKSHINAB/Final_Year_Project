# PRE-TRAINING AUDIT REPORT
## SOA-FasterRCNN — Research Experiment Readiness

**Audit Date:** 2026-08-22
**Project Root:** `c:\Users\prada\Downloads\Final Year Project\uavdet-system\`
**Audit Scope:** 20 mandatory checks before GPU training begins
**Automated Checks:** 24 sub-checks — 24 PASS / 0 FAIL
**Manual Review Findings:** 4 bugs (1 critical, 3 minor)

---

> **READ BUG-1 BEFORE RUNNING ANY EXPERIMENT.**
> It is architectural, not just cosmetic. Your decision changes what E2/E3/E4 claim to be.

---

## SUMMARY TABLE

| # | Check | Result | File:Line |
|---|-------|--------|-----------|
| 1 | E1 is a genuine standard Faster R-CNN + ResNet-50-FPN baseline | **PASS** | `baseline_fasterrcnn.py:104` |
| 2 | E2 changes ONLY the Coordinate Attention component | **PASS** | `proposed_model.py:150-162` |
| 3 | E3 changes ONLY the SACM/RPN component | **PASS** | `proposed_model.py:164-178` |
| 4 | E4 combines E2 and E3 without unintended changes | **PASS** | `proposed_model.py:131-180` |
| 5 | VisDrone train/val annotation loading | **PASS*** | `visdrone.py:69-78` |
| 6 | All 10 VisDrone classes and category mappings | **PASS** | 3 files consistent |
| 7 | Bounding boxes and image dimensions | **PASS** | `visdrone.py:84-123` |
| 8 | AP_small/medium/large via proper COCO evaluation | **PASS** | `metrics.py:111-116` |
| 9 | mAP@50 and mAP@50:95 | **PASS** | `metrics.py:142-149` |
| 10 | Precision and Recall | **PASS** | `metrics.py:175-204` |
| 11 | NMS and confidence thresholds | **PASS** | `fasterrcnn_visdrone.yaml:51-53` |
| 12 | Training/validation split | **PASS** | `visdrone.py:208-209` |
| 13 | No validation data leaks into training | **PASS** | `visdrone.py:211-215` |
| 14 | AMP/GPU training | **PASS** | `trainer.py:27-28, 250-259` |
| 15 | Checkpoint saving and resume | **PASS** | `trainer.py:310-323` |
| 16 | Parameter count for every experiment | **PASS** | Verified at runtime |
| 17 | All four experiments use identical training settings | **PASS** | Single shared YAML |
| 18 | Debug mode and research/full mode clearly separated | **PASS** | `fasterrcnn_visdrone.yaml:25-36` |
| 19 | No dummy metrics (AP=0.5 etc.) anywhere | **PASS** | All .py + .csv files |
| 20 | No fake or hard-coded experimental results | **PASS** | All .py + .csv files |

*\* Check 5: Dataset not yet on disk. Loader code is correct.*

---

## DETAILED FINDINGS

### CHECK 1 — E1: Genuine Faster R-CNN Baseline

**Result: PASS**
**File:** `research/models/baseline_fasterrcnn.py` lines 104-115

E1 is constructed via `torchvision.models.detection.FasterRCNN` with:
- Backbone: `resnet_fpn_backbone("resnet50")` — standard torchvision ResNet-50 + FPN
- Anchor sizes: `((16,), (32,), (64,), (128,), (256,))` — correct for VisDrone sub-32px objects
- Aspect ratios: `(0.5, 1.0, 2.0, 3.0)` per level
- No attention module, no SACM — verified programmatically

Verified at runtime:
```
E1 has AttentionFPN:       False  (expected False)  PASS
E1 has ScaleAwareRPNHead:  False  (expected False)  PASS
E1 params: 41.35M total / 41.12M trainable
```

---

### CHECK 2 — E2: Only Coordinate Attention Added

**Result: PASS**
**File:** `research/models/proposed_model.py` lines 150-162

E2 wraps `model.backbone.fpn` in `AttentionFPN`. No other change.
- `model.rpn.head` remains standard RPNHead (verified)
- `model.roi_heads` is unchanged (verified)
- `AttentionFPN` wraps the inner FPN and contains `.fpn` + `.attention` attributes

**However, see BUG-1 below for a critical issue with WHERE the attention is applied.**

---

### CHECK 3 — E3: Only SACM/RPN Added

**Result: PASS**
**File:** `research/models/proposed_model.py` lines 164-178

E3 replaces `model.rpn.head` with `ScaleAwareRPNHead`. No other change.
- `model.backbone.fpn` is the standard FPN (not wrapped) — verified
- `ScaleAwareRPNHead.forward(features)` signature matches torchvision's `RPNHead`

---

### CHECK 4 — E4: E2 + E3 Combined Without Unintended Changes

**Result: PASS**
**File:** `research/models/proposed_model.py`

Verified at runtime:
```
E4 has AttentionFPN:      True   PASS
E4 has ScaleAwareRPNHead: True   PASS
E4 params: 41.60M (= E1 + CA delta + SACM delta)  PASS
```

---

### CHECK 5 — VisDrone Annotation Loading

**Result: PASS*** (Code-level; dataset not yet on disk)
**File:** `research/datasets/visdrone.py` lines 69-78

- Reads `annotations/instances_{split}.json` — standard COCO format path
- Uses `pycocotools.COCO` API — standard, correct
- Raises `FileNotFoundError` with clear instructions if annotation file missing
- Dataset root from config: `training/datasets/uav/`

---

### CHECK 6 — Class Names Consistency

**Result: PASS**

All three files define the same 10 foreground classes in the same order:
```
pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
```

| Source | Background included | Foreground count | Indexing |
|--------|--------------------|--------------------|----------|
| `baseline_fasterrcnn.py` | Yes (`__background__`) | 10 | 1-10 |
| `metrics.py` | No | 10 | 1-10 |
| `visdrone.py` (CLASS_TO_IDX) | No | 10 | 1-10 |

Verified: `pedestrian=1`, `motor=10` — correct 1-indexed COCO format.

---

### CHECK 7 — Bounding Boxes and Image Dimensions

**Result: PASS**
**File:** `research/datasets/visdrone.py` lines 84-123

- Input from COCO JSON: XYWH format
- Converted to XYXY: `x2 = x + w, y2 = y + h`
- Clamped to image boundaries: `x1 = max(x,0)`, `x2 = min(x+w, width)`
- Degenerate filter: `(x2-x1) <= 1 or (y2-y1) <= 1` — dropped
- Area filter: `area < 4.0 px²` — dropped
- Output: XYXY float32 (required by Faster R-CNN)

See BUG-3 for a minor issue with empty-image handling.

---

### CHECK 8 — AP_small / AP_medium / AP_large

**Result: PASS**
**File:** `research/evaluation/metrics.py` lines 111-116

Uses `pycocotools.COCOeval` — the standard library for detection benchmarks.

Area thresholds (COCO standard, internal to COCOeval):
```
AP_small:  area < 1024 px²     (< 32×32 px)
AP_medium: 1024 <= area < 9216  (32×32 – 96×96 px)
AP_large:  area >= 9216         (> 96×96 px)
```

`stats[3]`, `stats[4]`, `stats[5]` map to AP_small, AP_medium, AP_large — correct.

---

### CHECK 9 — mAP@50 and mAP@50:95

**Result: PASS**
**File:** `research/evaluation/metrics.py` lines 142-149

```python
"map50_95": float(stats[0]),   # AP @ IoU 0.50:0.05:0.95
"map50":    float(stats[1]),   # AP @ IoU 0.50
"ap_small": float(stats[3]),   # AP for area < 1024 px²
```

COCO stats array indices are standard and verified.

---

### CHECK 10 — Precision and Recall

**Result: PASS**
**File:** `research/evaluation/metrics.py` lines 175-204

- Precision extracted at IoU=0.5 — `prec[0, :, :, 0, 2]` (index 0 = IoU=0.5)
- Recall extracted at IoU=0.5 — `eval_obj.eval["recall"][0, :, 0, 2]`
- F1 = harmonic mean of P and R

Note: These are averages over all categories and recall points — consistent with standard detection paper reporting.

---

### CHECK 11 — NMS and Confidence Thresholds

**Result: PASS**
**Files:** `fasterrcnn_visdrone.yaml` lines 51-53, `baseline_fasterrcnn.py` lines 108-114

| Parameter | Config Value | Applied to Model | Verified |
|-----------|------------|-----------------|---------|
| `box_nms_thresh` | 0.5 | `model.roi_heads.nms_thresh` | Yes |
| `box_score_thresh` | 0.05 | `model.roi_heads.score_thresh` | Yes |
| `rpn_nms_thresh` | 0.7 | `model.rpn.nms_thresh` | Yes |
| `rpn_pre_nms_top_n_train` | 4000 | Applied | Yes |
| `rpn_post_nms_top_n_train` | 2000 | Applied | Yes |

`box_score_thresh=0.05` is deliberately low for maximum recall during evaluation.

---

### CHECK 12 — Training/Validation Split

**Result: PASS**
**File:** `research/datasets/visdrone.py` lines 208-209

```python
train_ds = VisDroneDataset(dataset_root, "train", ...)
val_ds   = VisDroneDataset(dataset_root, "val", ...)
```

Separate instances reading from:
- `annotations/instances_train.json`
- `annotations/instances_val.json`

---

### CHECK 13 — No Validation Data Leaked into Training

**Result: PASS**
**File:** `research/datasets/visdrone.py` lines 211-215

- Subset sampling (debug mode) applied ONLY to `train_ds` — val is never subsampled
- No shared indices between train and val
- `shuffle=True` for train, `shuffle=False` for val

---

### CHECK 14 — AMP / GPU Training

**Result: PASS**
**File:** `research/training/trainer.py` lines 27-28, 250-259

- `GradScaler` and `autocast` used correctly
- AMP enabled only when `device.type == "cuda"` and `cfg["amp"] == True`
- Correct sequence: `scale() -> backward() -> unscale_() -> clip_grad -> step() -> update()`
- Gradient clipped AFTER `unscale_()` — correct per PyTorch docs

NOTE: `from torch.cuda.amp import GradScaler, autocast` is the legacy import path.
On PyTorch 2.0+ the preferred form is `torch.amp.GradScaler("cuda")`.
The legacy form still works but may print a deprecation warning. Not a training blocker.

---

### CHECK 15 — Checkpoint Saving and Resume

**Result: PASS**
**Files:** `trainer.py` lines 310-323, `baseline_fasterrcnn.py` lines 157-168

Saving:
```python
state = {
    "epoch": epoch, "model": model.state_dict(),
    "optimizer": optimizer.state_dict(), "metrics": metrics,
    "model_name": model_name
}
torch.save(state, "last_model.pth")   # every epoch
torch.save(state, "best_model.pth")   # when mAP50 improves
```

Loading (partial — handles shape mismatches for cross-experiment resume):
```python
matched = {k: v for k, v in state_dict.items()
           if k in own and own[k].shape == v.shape}
model.load_state_dict(own)
```

---

### CHECK 16 — Parameter Counts (All Four Experiments)

**Result: PASS**
**Verified at runtime:**

| Experiment | Model Name | Total Params | Trainable | Delta vs E1 |
|-----------|-----------|-------------|-----------|-------------|
| E1 | `baseline_fasterrcnn` | **41.35 M** | 41.12 M | — (reference) |
| E2 | `fasterrcnn_ca` | **41.36 M** | 41.14 M | +0.01 M |
| E3 | `fasterrcnn_sacm` | **41.58 M** | 41.36 M | +0.23 M |
| E4 | `fasterrcnn_ca_sacm` | **41.60 M** | 41.37 M | +0.25 M |

Ordering `E1 < E2 < E4` and `E1 < E3 < E4` — correct and verified.

GFLOPs: **NOT YET VERIFIED** — `fvcore` not installed.
Run `pip install fvcore` then re-run `get_model_complexity()` for publication-grade FLOPs.

---

### CHECK 17 — Identical Training Settings Across All Experiments

**Result: PASS**
**File:** `research/configs/fasterrcnn_visdrone.yaml`

A single shared YAML for all four experiments. Differences between experiments are ONLY
the `use_attention` and `use_sacm` flags in `build_proposed_model()`.

| Setting | Value (all experiments) |
|---------|------------------------|
| Optimizer | SGD, lr=0.005, momentum=0.9, wd=0.0001, nesterov=True |
| LR Schedule | Cosine annealing + 3-epoch warmup |
| Epochs (research) | 100 |
| Batch size | 4 |
| Image size | 1280 px |
| Seed | 42 |
| AMP | True |
| Grad clip | 10.0 |
| Backbone freeze | First 5 epochs |
| Backbone | ResNet-50, IMAGENET1K_V2 |
| Anchors | 16, 32, 64, 128, 256 px |
| Aspect ratios | 0.5, 1.0, 2.0, 3.0 |

---

### CHECK 18 — Debug vs Research Mode Separation

**Result: PASS**
**File:** `research/configs/fasterrcnn_visdrone.yaml` lines 25-36

| Parameter | Debug Mode | Research Mode |
|-----------|-----------|--------------|
| `img_size` | 640 | 1280 |
| `epochs` | 5 | 100 |
| `batch_size` | 2 | 4 |
| `subset_fraction` | 0.05 (5% of train) | 1.0 (100%) |
| `workers` | 2 | 8 |

Activated via `--mode debug` or `--mode research` CLI argument.

---

### CHECK 19 — No Dummy Metrics

**Result: PASS**

Automated scan of all Python files and CSVs:
- 0 instances of `ap = 0.5`, `map50 = 0.5`, `return 0.5` found
- 0 numeric `0.5` values in any CSV data rows
- `_empty_metrics()` returns all-zeros, not fabricated values

---

### CHECK 20 — No Fake Experimental Results

**Result: PASS**

- `ABLATION_RESULTS.csv`: all metric fields contain `"NOT YET VERIFIED"`
- `ERROR_ANALYSIS.md`: all values are `"[TBF]"` placeholders
- `PAPER_NOTES.md`: all results are `"[RESULT TO BE FILLED AFTER TRAINING]"`
- `RESEARCH_STATUS.md`: explicitly states all metrics as `NOT YET VERIFIED`
- No training has been performed — no results exist anywhere in the repo

---

## BUGS FOUND

---

### BUG-1 — CRITICAL: Attention Applied at FPN OUTPUT, Not Lateral Connections

**Severity:** CRITICAL (architectural mislabel — changes the research claim)
**File:** `research/models/proposed_model.py` lines 92-100

**Exact code:**
```python
def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Run standard FPN
    out = self.fpn(x)
    # Apply attention to specified levels of FPN output
    # (In this implementation, attention is applied post-FPN output
    #  as a practical approximation. True lateral-connection insertion
    #  requires modifying torchvision internals.)
    out = self.attention(out)
    return out
```

**What the code actually does:**
Coordinate Attention is applied to P2 and P3 AFTER the full FPN top-down fusion.
This is "CA at FPN output levels P2/P3."

**What every document claims:**
`RESEARCH_GAP.md`, `NOVELTY_CLAIM.md`, `PAPER_NOTES.md`, and the module docstring all state:
"Coordinate Attention at FPN LATERAL CONNECTIONS (P2/P3), applied BEFORE top-down fusion."

**Why this matters for the paper:**
The claimed novelty is that CA is inserted at the lateral connection (pre-fusion) rather than at
the FPN output (post-fusion). If it is post-fusion, then:
- The novelty argument changes from "pre-fusion spatial calibration" to "post-fusion recalibration"
- The comparison "existing work applies attention at FPN output (our claimed weakness)" now
  applies to our own method
- The scientific distinction is real and affects the paper's contribution section

**Two valid options — you must choose one:**

OPTION A: Fix the description (keep current code)
  Update all documentation to say "CA at FPN output P2/P3" instead of "lateral connections."
  The model implementation is unchanged. The novelty claim becomes:
  "Scale-selective CA applied only to P2/P3 FPN output levels within Faster R-CNN on VisDrone"
  This is still a valid, defensible contribution — just with a different framing.

OPTION B: Fix the code (fulfils original design intent)
  Override `FeaturePyramidNetwork.forward()` in a subclass to insert attention
  at the lateral connection — specifically before the element-wise `+` (top-down + lateral).
  This is ~40 lines of code. The architecture remains Faster R-CNN. No training changes needed.

RECOMMENDATION: Option A if you want to start E2/E3/E4 immediately after E1.
                Option B if you want the implementation to match the stated novelty exactly.

This is NOT a training-blocker for E1. E1 can start now.
You MUST resolve BUG-1 before running E2, E3, or E4 to avoid publishing wrong architecture claims.

---

### BUG-2 — MINOR: Scheduler Resets at Backbone Unfreeze

**Severity:** MINOR (affects LR curve shape; does not break training if consistent across all experiments)
**File:** `research/training/trainer.py` lines 235-238

**Exact code:**
```python
# In train_epoch():
if epoch == freeze_epochs + 1:
    for p in self.model.backbone.parameters():
        p.requires_grad = True
    self._build_optimizer()  # creates a NEW WarmupCosineScheduler(current_epoch=0)
```

**Effect:**
At epoch 6 (backbone unfreeze), `_build_optimizer()` creates a new `WarmupCosineScheduler`
with `current_epoch=0`. In the next loop iteration, `self.scheduler.step()` advances it to
epoch 1 — warmup phase. The LR re-ramps from `0.0005` to `0.005` over 3 epochs.

This is a re-warmup after backbone unfreeze, which may or may not be intentional.

**Does it break ablation fairness?**
No — the same code runs for all 4 experiments. The LR schedule is identical across E1–E4.

**Fix (if re-warmup is unintended):**
After `_build_optimizer()`, set `self.scheduler.current_epoch = epoch` to continue decay.

**For the paper:** Document the LR behaviour:
"After backbone unfreeze at epoch 6, LR re-warms from 0.0005 to 0.005 over 3 epochs
before resuming cosine decay to 5e-5."

---

### BUG-3 — MINOR: Empty Image Dummy Box Uses Label=0 (Background)

**Severity:** MINOR (rare edge case; does not block training)
**File:** `research/datasets/visdrone.py` lines 110-115

**Exact code:**
```python
if not boxes:
    boxes   = [[0.0, 0.0, 1.0, 1.0]]
    labels  = [0]    # label 0 = background class
    areas   = [1.0]
    iscrowd = [0]
```

**Effect:**
If all annotations in an image are smaller than 2x2px (after filtering), a dummy 1x1 box
with label=0 (background) is inserted. The RPN may assign a spurious objectness label
to the nearest anchor. In practice this case is very rare on VisDrone.

**Correct fix:**
```python
if not boxes:
    boxes   = torch.zeros((0, 4), dtype=torch.float32)
    labels  = torch.zeros((0,),   dtype=torch.int64)
    areas   = torch.zeros((0,),   dtype=torch.float32)
    iscrowd = torch.zeros((0,),   dtype=torch.int64)
```

Verify that your torchvision version handles empty targets before applying (test with --mode debug).

---

### BUG-4 — MINOR: pycocotools Missing from requirements.txt

**Severity:** MINOR (install blocker; simple fix)
**File:** `requirements.txt`

`pycocotools` is imported by `metrics.py` and `visdrone.py` but not listed in `requirements.txt`.
Caused `ModuleNotFoundError` during initial audit run.

**Fix:** Add to `requirements.txt`:
```
pycocotools>=2.0.7
```

pycocotools 2.0.11 is now installed on this machine. Fix the file before uploading to HPC.

---

## FIXES SUMMARY

| Bug | Severity | Action Required | Blocks Training? |
|-----|---------|----------------|-----------------|
| BUG-1: Attention at FPN output vs lateral | CRITICAL | Choose Option A or B | E2/E3/E4 only |
| BUG-2: Scheduler reset at backbone unfreeze | MINOR | Document in paper | No |
| BUG-3: Empty image dummy box label=0 | MINOR | Fix before full training | No (rare) |
| BUG-4: pycocotools missing from requirements | MINOR | Add one line | No (already installed) |

---

## GO / NO-GO DECISION

```
E1 (Baseline Faster R-CNN):      GO  — start immediately
E2/E3/E4 (Proposed Models):      CONDITIONAL HOLD
  Reason: BUG-1 must be resolved first.
  Once you choose Option A or B, E2/E3/E4 can run.
```

All training infrastructure is correct:
- AMP, checkpointing, gradient clipping: correct
- COCO metrics (mAP50, mAP50:95, AP_small, AP_medium, AP_large): correct
- Class mappings and 1-indexing: correct
- No data leaks, no fake results: confirmed
- Identical training settings across all 4 experiments: confirmed

---

## FIRST COMMAND TO RUN

### Step 1: Debug sanity check (run this first — ~10 minutes on GPU)

```bash
python research/run_experiment.py --experiment baseline --mode debug --device cuda
```

Expected:
- Loss decreases over 5 epochs
- `experiments/baseline/metrics.json` produced with `map50`, `ap_small` values
- No CUDA errors or shape mismatches

### Step 2: Full baseline training

```bash
python research/run_experiment.py --experiment baseline --mode research --device cuda
```

Outputs saved to:
- `experiments/baseline/weights/best_model.pth`
- `experiments/baseline/logs/training_log.csv`
- `experiments/baseline/metrics.json`

### Step 3 (After BUG-1 resolved): Run E2, E3, E4

```bash
python research/run_experiment.py --experiment attention      --mode research --device cuda
python research/run_experiment.py --experiment second_module  --mode research --device cuda
python research/run_experiment.py --experiment final_model    --mode research --device cuda
```

---

## REPLY REQUIRED

Reply with which BUG-1 option you choose:
- **"Option A"** — update the description, keep the code (CA at FPN output P2/P3)
- **"Option B"** — fix the code to truly insert CA at lateral connections (pre-fusion)

The fix will then be applied before E2/E3/E4 training begins.

---

*Audit completed 2026-08-22. No experimental results were fabricated. No novelty was claimed.
No publication readiness was implied. All unverified values remain marked NOT YET VERIFIED.*
