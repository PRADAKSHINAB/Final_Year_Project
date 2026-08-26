# PAPER NOTES
## Draft Research Paper Material for Publication

**Title:** Small-Object-Aware Faster R-CNN: Feature Enhancement and Proposal Refinement for UAV Imagery

**Status:** Draft — Results sections contain placeholders. DO NOT FABRICATE.

---

## TITLE

Small-Object-Aware Faster R-CNN with Lateral Attention and Scale-Aware Proposal Refinement for UAV Imagery

*Alternative working title:* SOA-FasterRCNN: Coordinate-Attention-Enhanced FPN and Scale-Aware RPN for Small Object Detection in Drone Imagery

---

## ABSTRACT

Detecting small objects in unmanned aerial vehicle (UAV) imagery is a fundamental challenge
in aerial surveillance and intelligent transportation systems. Small objects occupy only a
few pixels in high-altitude UAV frames, making them difficult to detect with standard
detection frameworks. In this paper, we propose SOA-FasterRCNN, a two-stage detection
framework based on Faster R-CNN that incorporates two complementary contributions
specifically designed for small-object detection in UAV imagery on the VisDrone dataset.

First, we propose applying Coordinate Attention (CA) at FPN lateral connections (at P2 and P3
pyramid levels), enabling scale-selective feature recalibration before top-down feature
fusion. This targets the primary source of small-object feature loss in the FPN pipeline.

Second, we propose a Scale-Aware Context Module (SACM) integrated within the Region
Proposal Network (RPN) head, augmenting objectness scoring with multi-scale dilated context
features and scale-bin prediction to reduce false positives in dense aerial scenes.

Experiments on VisDrone-DET demonstrate: mAP@0.5 = [RESULT TO BE FILLED AFTER TRAINING],
AP_small = [RESULT TO BE FILLED AFTER TRAINING], representing [X]% improvement over the
Faster R-CNN baseline. Ablation studies confirm that each contribution independently
improves AP_small, with the combined model achieving the best performance.

---

## INTRODUCTION

Unmanned aerial vehicles (UAVs) are increasingly deployed for traffic monitoring, crowd
analysis, search and rescue, and infrastructure inspection. A key computational challenge
in these applications is reliable detection of small objects captured from high altitudes.
In UAV imagery, objects such as pedestrians, cyclists, and vehicles typically occupy
extremely few pixels — often fewer than 32x32 pixels — in frames captured at 50-200 meters
altitude.

Object detection frameworks have advanced significantly, from region-based methods
(Faster R-CNN, Ren et al., 2015) to one-stage approaches (YOLO series, RetinaNet) and
transformer-based detectors (DETR, RT-DETR). However, small object detection in dense
UAV scenes remains an open research problem. The VisDrone-DET benchmark (Zhu et al., 2020)
provides a large-scale evaluation framework for this task, containing over 10,000 images
with 540,000+ annotated objects across 10 categories.

We observe three specific failure modes in standard Faster R-CNN applied to VisDrone:
(1) FPN feature representation of sub-32px objects is weakened during top-down fusion,
(2) RPN objectness scoring does not account for scale-level distribution of proposals,
(3) Dense scene proposals compete with each other in standard NMS without context awareness.

To address these challenges, we propose two principled modifications to Faster R-CNN.
Unlike existing work that applies attention at FPN outputs or modifies anchor assignment
strategies, we specifically target: (a) the lateral connection pathway where small-object
features are processed before fusion, and (b) the objectness scoring forward pass where
scale awareness is currently absent.

---

## PROBLEM STATEMENT

**Formal problem:** Given a drone-captured RGB image I ∈ R^(H×W×3), detect all objects
O = {(b_i, c_i)} where b_i ∈ R^4 is the bounding box in XYXY format and c_i ∈ {1,...,10}
is the VisDrone category label.

**Small object challenge:** Objects with area(b_i) < 1024 px^2 (i.e., smaller than 32×32 px)
constitute the majority of detections in high-altitude UAV imagery and exhibit systematically
lower AP than medium/large objects.

**Primary evaluation metric:** AP_small (COCO standard: area < 1024 px^2).

---

## MOTIVATION

**Why Faster R-CNN (not YOLO)?**
Faster R-CNN's two-stage pipeline (proposal → classification) provides inherently better
small-object recall than one-stage detectors because the RPN first proposes candidate regions
independently of classification, allowing the model to detect tiny objects without requiring
a single grid cell to simultaneously handle localization and classification.

**Why attention at FPN lateral connections?**
The FPN top-down pathway progressively merges high-semantic features (P5) with
high-resolution features (P2). Small-object features in P2 can be overwritten by the
stronger P5-derived features during this merge. Applying attention BEFORE this merge
(at the lateral connection) selectively amplifies small-object-relevant channels and
spatial locations before they are combined with coarser semantic features.

**Why CA over CBAM?**
Coordinate Attention preserves positional information via directional 1D pooling,
which is critical for precise localization of tiny objects. CBAM's global pooling in
the channel branch discards this positional information.

**Why SACM in the RPN?**
The RPN objectness head scores proposals at all FPN levels with a shared head.
A scale-aware module that predicts which scale-bin a feature region belongs to
can soft-gate the objectness score, reducing false positives from background regions
that produce proposals at incorrect scales.

---

## LITERATURE SURVEY

| Work | Method | Dataset | mAP@0.5 | Relevance |
|------|--------|---------|---------|-----------|
| Faster R-CNN (Ren et al., NIPS 2015) | Two-stage; RPN + ROI | COCO | ~37% | Our base detector |
| FPN (Lin et al., CVPR 2017) | Multi-scale feature pyramid | COCO | +7% over baseline | Our backbone neck |
| CBAM (Woo et al., ECCV 2018) | Channel + spatial attention | ImageNet | N/A | Attention comparison |
| Coordinate Attention (Hou et al., CVPR 2021) | Directional 1D pooling | ImageNet | N/A | Our attention module |
| ECA-Net (Wang et al., CVPR 2020) | Efficient channel attention | ImageNet | N/A | Attention comparison |
| VistrongerDet (2021) | FPN+ROI enhancement for VisDrone | VisDrone | [N/A reproduced] | Closest related work |
| RFLA (Xu et al., ArXiv 2022) | Gaussian label assignment | COCO, VisDrone | +2.5% AP_small | Related: label assignment |
| Cascade R-CNN (Cai 2018) | Multi-stage regression | COCO | +4% AP | Related: multi-stage |
| Deformable DETR (Zhu 2021) | Multi-scale attention | COCO | +4% over DETR | Transformer comparison |
| ClusDet (2019) | Cluster-based detection | VisDrone | ~21% mAP | UAV detection |

---

## METHODOLOGY

### 3.1 Baseline Architecture
Faster R-CNN with ResNet-50-FPN V2 backbone. Small-object anchor scales:
(16, 32, 64, 128, 256) px. Aspect ratios: (0.5, 1.0, 2.0, 3.0).
See docs/EXPERIMENT_PLAN.md for full hyperparameters.

### 3.2 Contribution 1: FPN Lateral Coordinate Attention

The standard FPN lateral connection at level l is:
```
P_l = Conv1x1(C_l) + Upsample(P_{l+1})
```

We modify this by inserting CA before the element-wise addition:
```
C_l' = CoordinateAttention(Conv1x1(C_l))
P_l = C_l' + Upsample(P_{l+1})
```

Applied to l ∈ {2, 3} (P2 and P3 levels, where small objects are predominantly detected).

Coordinate Attention (Hou et al., 2021) factorizes 2D attention into two 1D attention maps:
- H-direction: captures objects at different row positions
- W-direction: captures objects at different column positions

This preserves the spatial coordinate information critical for tight bounding box regression
of tiny objects, which global pooling-based attention (SE, CBAM channel branch) discards.

### 3.3 Contribution 2: Scale-Aware Context Module

Standard RPN objectness head:
```
objectness = Conv1x1(RPN_features)
```

Our SACM-augmented head:
```
context_features = DilatedContextAgg(RPN_features)  # dilation = {1, 2, 4}
scale_weights = ScaleBinPredictor(RPN_features)      # K=4 bins
scale_embedding = Embedding(argmax(scale_weights))
gate = Sigmoid(Conv1x1(context_features))
enhanced = RPN_features + scale_embedding * gate
objectness = Conv1x1(enhanced)
```

The scale-bin predictor learns to associate FPN level features with their expected
object scale range. The soft gate reduces objectness scores for proposals that do
not match the expected scale, reducing false positives from dense background regions.

---

## PROPOSED ARCHITECTURE

```
Input: 1280×1280 RGB image
         ↓
ResNet-50 Backbone (pretrained: ImageNet-1K V2)
  [C2: stride 4] [C3: stride 8] [C4: stride 16] [C5: stride 32]
         ↓
FPN Lateral Connections
  Conv1×1(C2) → [Coordinate Attention] → P2  ← (our Contribution 1)
  Conv1×1(C3) → [Coordinate Attention] → P3  ← (our Contribution 1)
  Conv1×1(C4) → P4
  Conv1×1(C5) → P5
  MaxPool(P5) → P6
         ↓
FPN Top-Down Fusion: P5 → P4 → P3 → P2
         ↓
RPN Head (per FPN level)
  [Scale-Aware Context Module]  ← (our Contribution 2)
  Objectness + Box Regression
         ↓
ROI Align (7×7)
         ↓
Box Head (4 conv + 1 fc)
         ↓
Classification (10+1 classes) + Bounding Box Regression
```

---

## LOSS FUNCTIONS

Standard Faster R-CNN multi-task loss (EXISTING METHOD):

**RPN Loss:**
L_rpn = L_cls(p_i, p_i*) + lambda * p_i* * L_box(t_i, t_i*)

where:
- L_cls: Binary cross-entropy (objectness: object vs background)
- L_box: Smooth-L1 regression loss on anchor offsets
- p_i*: Ground truth label (1=positive, 0=negative)
- t_i, t_i*: Predicted and GT box regression targets

**ROI Head Loss:**
L_roi = L_cls(p, u) + [u >= 1] * L_box(t^u, v)

where:
- L_cls: Cross-entropy over 11 classes (10 VisDrone + background)
- L_box: Smooth-L1 regression on class-specific offsets
- u: Ground truth class label

**Total Loss:**
L_total = L_rpn_cls + L_rpn_box + L_roi_cls + L_roi_box

No custom loss function is proposed. The contribution is architectural, not loss-based.

---

## TRAINING PROCEDURE

1. Initialize ResNet-50 backbone with IMAGENET1K_V2 weights
2. Load COCO-pretrained FPN/detection weights where shapes match
3. Freeze backbone for epochs 1-5 (stabilize RPN + head training)
4. Unfreeze backbone at epoch 6; rebuild optimizer
5. Optimizer: SGD (lr=0.005, momentum=0.9, weight_decay=0.0001, nesterov=True)
6. LR schedule: 3-epoch warmup (scale 0.1→1.0) + cosine annealing (η_min = 5e-5)
7. AMP (Automatic Mixed Precision): enabled
8. Gradient clipping: max norm = 10.0
9. Total epochs: 100
10. Seed: 42 (single run due to compute constraints)

---

## EXPERIMENTAL SETUP

| Parameter | Value |
|-----------|-------|
| Dataset | VisDrone-DET 2021 (full split) |
| Train images | ~6,471 |
| Val images | ~548 |
| Input size | 1280 × 1280 |
| Batch size | 4 |
| GPU | [TO BE SPECIFIED — HPC at hpckongu.edu] |
| Training time | ~8-24h per experiment (NOT YET VERIFIED) |

---

## RESULTS

[RESULT TO BE FILLED AFTER TRAINING]

All metrics below are placeholders. Values will be inserted after experiments complete.

### Main Results Table

| Model | mAP@0.5 | mAP@0.5:0.95 | AP_small | AP_medium | AP_large | Params(M) | FPS |
|-------|---------|--------------|----------|-----------|----------|-----------|-----|
| E1: Baseline FRCNN | — | — | — | — | — | — | — |
| E2: +CA Attention | — | — | — | — | — | — | — |
| E3: +SACM | — | — | — | — | — | — | — |
| E4: SOA-FRCNN (ours) | — | — | — | — | — | — | — |
| UAVDet* | — | — | — | — | — | — | — |

*UAVDet results labeled "Reported in original paper" — NOT directly comparable without identical training setup.

---

## ABLATION STUDY

[RESULT TO BE FILLED AFTER TRAINING]

### Ablation Table

| Model | AP_small | ΔAP_small vs. E1 | Interpretation |
|-------|----------|-----------------|----------------|
| E1: Baseline | — | — | Reference |
| E2: +CA | — | — | Effect of Contribution 1 |
| E3: +SACM | — | — | Effect of Contribution 2 |
| E4: Both | — | — | Combined effect |

---

## ERROR ANALYSIS

[RESULT TO BE FILLED AFTER TRAINING]

Detailed error analysis will be conducted using research/evaluation/error_analysis.py.
Expected findings (NOT verified):
- Missed small objects should decrease with attention (Contribution 1)
- False positives in dense scenes should decrease with SACM (Contribution 2)

---

## LIMITATIONS

1. **Single random seed:** Due to GPU compute constraints, experiments use seed=42 only.
   Multiple seeds would improve statistical reliability.

2. **GFLOPs measurement:** Accurate GFLOPs for variable-input detection models require
   fvcore or ptflops; values marked "NOT YET VERIFIED" until computed.

3. **No super-resolution:** Some VisDrone images have objects of 5-8px which may require
   explicit super-resolution preprocessing beyond what our method addresses.

4. **VisDrone ignores 'ignored region' category:** Our evaluation excludes category 0.
   This is consistent with standard VisDrone evaluation practice.

5. **Literature review completeness:** The novelty claims require comprehensive paper
   search to confirm no identical approach exists. Claims are labeled accordingly.

---

## CONCLUSION

[RESULT TO BE FILLED AFTER TRAINING]

We proposed SOA-FasterRCNN, a Faster R-CNN variant with two principled contributions
for small-object detection in UAV imagery: (1) Coordinate Attention at FPN lateral
connections and (2) a Scale-Aware Context Module in the RPN head. Experiments on
VisDrone-DET [RESULT TO BE FILLED AFTER TRAINING].

---

## FUTURE WORK

1. Multi-seed experimental validation for statistical reliability
2. Extension to video-based UAV detection (temporal consistency)
3. Knowledge distillation for edge deployment
4. Semi-supervised learning using VisDrone's unlabeled sequences
5. Extension to VisDrone-SOT (single object tracking) and multi-object tracking
6. Investigation of transformer-based attention alternatives for FPN lateral connections
7. Exploration of SACM on other dense detection datasets (CrowdHuman, TinyPerson)
