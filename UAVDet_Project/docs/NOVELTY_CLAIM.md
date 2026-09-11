# Novelty Claim Document
## Small-Object-Aware Faster R-CNN for UAV Imagery

**Status:** POTENTIAL NOVELTY — REQUIRES VERIFICATION  
**Date:** 2026-08-21  
**Critical Rule:** We do NOT claim that any individual component is novel.

---

## Important Disclaimer

This document carefully distinguishes between:
- **EXISTING METHOD**: Techniques already published and validated in literature
- **POTENTIAL NOVELTY — REQUIRES VERIFICATION**: Combinations/adaptations that may not have identical published precedent, but require thorough literature confirmation
- **NOT A VALID NOVELTY CLAIM**: Things that are clearly already known

We explicitly acknowledge:
- CBAM is NOT novel (Woo et al., ECCV 2018)
- FPN is NOT novel (Lin et al., CVPR 2017)
- ResNet-50 is NOT novel (He et al., CVPR 2016)
- Coordinate Attention is NOT novel (Hou et al., CVPR 2021)
- P2 feature level is NOT novel (widely used since 2018+)
- Faster R-CNN is NOT novel (Ren et al., NIPS 2015)

---

## Component-Level Novelty Analysis

### Contribution 1: Coordinate Attention at FPN Lateral Connections

| Aspect | Status | Evidence |
|--------|--------|---------|
| Coordinate Attention module itself | EXISTING METHOD | Hou et al., CVPR 2021 |
| FPN as a feature extraction neck | EXISTING METHOD | Lin et al., CVPR 2017 |
| Attention applied at FPN output | EXISTING METHOD | Multiple papers (2020-2024) |
| Attention in backbone only | EXISTING METHOD | SE, CBAM papers and derivatives |
| CA specifically at lateral connections (pre-fusion) | POTENTIAL NOVELTY — REQUIRES VERIFICATION | Limited precedent found in search |
| CA for P2/P3 specifically (scale-selective) | POTENTIAL NOVELTY — REQUIRES VERIFICATION | Level-specific attention is not standard |
| CA in Faster R-CNN (not YOLO) | POTENTIAL NOVELTY — REQUIRES VERIFICATION | Most CA-UAV papers use YOLO |
| CA for VisDrone sub-32px domain adaptation | POTENTIAL NOVELTY — REQUIRES VERIFICATION | Requires validation |

**What is actually claimed:**
The SPECIFIC COMBINATION of (Coordinate Attention) + (FPN lateral connections) + (P2/P3 level targeting) + (Faster R-CNN framework) + (VisDrone sub-32px tuning) has not been found in an identical form in our literature search.

**What is NOT claimed:**
- Any individual module is novel
- The general idea of attention in FPN is novel
- The general idea of improving small-object detection is novel

---

### Contribution 2: Scale-Aware Context Module (SACM)

| Aspect | Status | Evidence |
|--------|--------|---------|
| RPN objectness head | EXISTING METHOD | Ren et al., NIPS 2015 |
| Dilated convolutions | EXISTING METHOD | Widely used since 2016 |
| Multi-scale context features | EXISTING METHOD | ASPP, DeepLab, etc. |
| Scale-aware label assignment | EXISTING METHOD | RFLA (Xu et al., 2022) |
| Post-processing NMS adaptation | EXISTING METHOD | Adaptive NMS, Soft-NMS |
| Scale-bin prediction within RPN forward pass | POTENTIAL NOVELTY — REQUIRES VERIFICATION | Distinct from label assignment |
| Soft-gated objectness scoring (scale-conditioned) | POTENTIAL NOVELTY — REQUIRES VERIFICATION | No identical approach found |
| Applied to Faster R-CNN on VisDrone | POTENTIAL NOVELTY — REQUIRES VERIFICATION | Needs confirmation |

**What is actually claimed:**
Augmenting the RPN objectness scoring with a lightweight scale-prediction + soft-gating module WITHIN the forward pass (not at training label assignment, not at post-processing NMS) for dense UAV scene detection has not been found in identical form.

**What is NOT claimed:**
- Dilated convolutions are novel
- Scale-awareness is novel
- Context aggregation is novel

---

## Method Comparison Table

| Method | Existing? | UAV? | Faster RCNN? | VisDrone? | Our Modification |
|--------|-----------|------|--------------|-----------|-----------------|
| CBAM at FPN output | YES | YES | Partial | Partial | Different location (lateral vs output), different module (CA vs CBAM) |
| SE in backbone | YES | YES | YES | Partial | Different location (FPN, not backbone) |
| ECA at FPN output | YES | Partial | Partial | No | Different location + CA's positional encoding |
| Coordinate Attention in backbone | YES | Partial | No | No | Different location (FPN lateral), different framework (FRCNN), different target (sub-32px) |
| RFLA label assignment | YES | YES | YES | YES | Different stage (forward pass vs training), different mechanism (soft gate vs Gaussian) |
| Adaptive NMS | YES | Partial | YES | Partial | Different stage (within RPN vs post-processing) |
| VistrongerDet | YES | YES | YES | YES | Different approach (our focus: attention at lateral + SACM in RPN) |

---

## What the Paper Can Legitimately Claim

### Valid Claims (with experimental support):
1. "We propose a Coordinate Attention module inserted at FPN lateral connections, specifically targeting P2/P3 scale levels, for improving feature representation of sub-32px objects in Faster R-CNN on VisDrone."
2. "We propose a Scale-Aware Context Module in the RPN head that augments objectness scoring with scale-bin prediction and multi-scale dilated context features for dense UAV scenes."
3. "Experimental results demonstrate [X]% improvement in AP_small compared to the Faster R-CNN baseline." [RESULT TO BE FILLED AFTER TRAINING]
4. "The ablation study confirms that each contribution independently improves AP_small." [RESULT TO BE FILLED AFTER TRAINING]

### Invalid Claims (must NOT appear in paper):
- "CBAM is proposed in this paper" — FALSE
- "FPN is a novel contribution" — FALSE
- "Coordinate Attention is our novel module" — FALSE (Hou et al., 2021)
- "Attention mechanisms have never been used for UAV detection" — FALSE
- "Our method achieves state-of-the-art on VisDrone" — Requires experimental verification

---

## Where Novelty Actually Resides

The scientific contribution, if validated by experiments, is:

1. **A specific architectural adaptation**: The selection and placement of CA at FPN lateral connections (not standard locations) specifically for the small-object problem in two-stage detection
2. **A domain-specific design decision**: The targeting of P2/P3 specifically (rather than all FPN levels) based on the scale distribution analysis of VisDrone objects
3. **A complementary module design**: The SACM that addresses a different failure mode (proposal quality) compared to the attention module (feature quality), creating a principled multi-contribution framework
4. **Experimental validation**: Proving that this specific combination improves AP_small on VisDrone with Faster R-CNN as baseline

---

## Literature Cross-Check Required Before Submission

The following papers must be checked to verify no identical approach exists:
- [ ] All ECCV/CVPR/ICCV 2022-2024 papers with "Faster RCNN" + "UAV/aerial" + "attention"
- [ ] All VisDrone Challenge workshop papers (2018-2024)
- [ ] VistrongerDet variants and extensions
- [ ] Papers citing Coordinate Attention + detection
- [ ] Papers on "context-aware RPN" or "scale-aware RPN" in detection
- [ ] RFLA and related label assignment papers with forward-pass variants
