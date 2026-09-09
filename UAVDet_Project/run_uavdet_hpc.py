"""
run_uavdet_hpc.py  v4 (Verified & Tested)
=========================================
ALL-IN-ONE RESEARCH PIPELINE — VisDrone Small-Object Detection
Faster R-CNN + ResNet-50-FPN | Ablation Suite E1, E2, E3, E4, E5
"""
from __future__ import annotations

import argparse, csv, json, math, os, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.tv_tensors as tv_tensors
from torchvision.io import read_image
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import v2 as T

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VISDRONE_CLASSES = [
    "pedestrian","people","bicycle","car","van",
    "truck","tricycle","awning-tricycle","bus","motor",
]
VISDRONE_CLASS_NAMES = ["__background__"] + VISDRONE_CLASSES
VISDRONE_NUM_CLASSES  = 11

CLASS_COLORS = {
    1:(0,255,0), 2:(0,200,100), 3:(255,255,0), 4:(0,0,255), 5:(255,0,0),
    6:(0,165,255), 7:(255,0,255), 8:(180,0,180), 9:(0,255,255), 10:(200,200,0),
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER: GroupNorm that always works
# ─────────────────────────────────────────────────────────────────────────────
def safe_gn(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    g = preferred_groups
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────
class VisDroneDataset(Dataset):
    def __init__(self, root: Path, split: str,
                 transforms: Optional[Any] = None, min_box_area: float = 1.0):
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.min_box_area = min_box_area

        candidates = [
            self.root / "annotations" / f"instances_{split}.json",
            self.root / f"VisDrone2019-DET-{split}" / "annotations" / f"instances_{split}.json",
            Path("Dataset") / f"VisDrone2019-DET-{split}" / "annotations" / f"instances_{split}.json",
            Path("Dataset") / f"VisDrone2019-DET-{split}" /
                f"VisDrone2019-DET-{split}" / "annotations" / f"instances_{split}.json",
        ]
        ann_path = next((p for p in candidates if p.exists()), None)
        if ann_path is None:
            raise FileNotFoundError(
                f"Cannot find instances_{split}.json. Tried:\n" +
                "\n".join(str(c) for c in candidates))

        self.coco = COCO(str(ann_path))
        self.ids  = sorted(self.coco.getImgIds())

        img_cands = [
            self.root / "images",
            ann_path.parent.parent / "images",
        ]
        self._img_dir = next((p for p in img_cands if p.exists()), self.root / "images")

    def __len__(self): return len(self.ids)

    def _load_target(self, img_id, W, H):
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x,y,w,h = a["bbox"]
            x1,y1 = max(float(x),0.), max(float(y),0.)
            x2,y2 = min(float(x+w),float(W)), min(float(y+h),float(H))
            if (x2-x1) <= 0.5 or (y2-y1) <= 0.5: continue
            if (x2-x1)*(y2-y1) < self.min_box_area: continue
            boxes.append([x1,y1,x2,y2])
            labels.append(int(a["category_id"]))
            areas.append((x2-x1)*(y2-y1))
            iscrowd.append(int(a.get("iscrowd",0)))

        if not boxes:
            return {"boxes":torch.zeros((0,4),dtype=torch.float32),
                    "labels":torch.zeros((0,),dtype=torch.int64),
                    "image_id":torch.as_tensor([img_id],dtype=torch.int64),
                    "area":torch.zeros((0,),dtype=torch.float32),
                    "iscrowd":torch.zeros((0,),dtype=torch.int64)}

        return {"boxes":torch.as_tensor(boxes,dtype=torch.float32),
                "labels":torch.as_tensor(labels,dtype=torch.int64),
                "image_id":torch.as_tensor([img_id],dtype=torch.int64),
                "area":torch.as_tensor(areas,dtype=torch.float32),
                "iscrowd":torch.as_tensor(iscrowd,dtype=torch.int64)}

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        meta   = self.coco.loadImgs([img_id])[0]
        image  = read_image(str(self._img_dir / meta["file_name"]))
        _,H,W  = image.shape
        target = self._load_target(img_id, W, H)

        if self.transforms is not None:
            img_tv  = tv_tensors.Image(image)
            boxes_tv = tv_tensors.BoundingBoxes(
                target["boxes"], format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(H,W))
            sample = self.transforms({"image":img_tv,"boxes":boxes_tv,"labels":target["labels"]})
            img_out = sample["image"]
            if isinstance(img_out, tv_tensors.Image): img_out = img_out.data
            boxes_out  = sample["boxes"]
            if isinstance(boxes_out, tv_tensors.BoundingBoxes): boxes_out = boxes_out.data
            labels_out = sample["labels"]
            if img_out.dtype == torch.uint8: img_out = img_out.float() / 255.0

            if len(boxes_out) > 0:
                valid = (boxes_out[:,2]-boxes_out[:,0] > 0.5) & (boxes_out[:,3]-boxes_out[:,1] > 0.5)
                if valid.any():
                    boxes_out  = boxes_out[valid].float()
                    labels_out = labels_out[valid].long()
                    area_out   = (boxes_out[:,2]-boxes_out[:,0])*(boxes_out[:,3]-boxes_out[:,1])
                    iscrowd_out= torch.zeros(len(boxes_out),dtype=torch.int64)
                else:
                    boxes_out=torch.zeros((0,4),dtype=torch.float32)
                    labels_out=torch.zeros((0,),dtype=torch.int64)
                    area_out=torch.zeros((0,),dtype=torch.float32)
                    iscrowd_out=torch.zeros((0,),dtype=torch.int64)
            else:
                boxes_out=torch.zeros((0,4),dtype=torch.float32)
                labels_out=torch.zeros((0,),dtype=torch.int64)
                area_out=torch.zeros((0,),dtype=torch.float32)
                iscrowd_out=torch.zeros((0,),dtype=torch.int64)

            return img_out, {"boxes":boxes_out,"labels":labels_out,
                             "image_id":target["image_id"],
                             "area":area_out,"iscrowd":iscrowd_out}

        return image.float()/255.0, target


def get_train_transforms(img_size):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomPhotometricDistort(p=0.8),
        T.RandomZoomOut(fill={tv_tensors.Image:(0,0,0)}, side_range=(1.0,2.5), p=0.3),
        T.RandomIoUCrop(min_scale=0.4, max_scale=1.0, min_aspect_ratio=0.5),
        T.SanitizeBoundingBoxes(min_size=1.0),
        T.Resize((img_size,img_size), antialias=True),
        T.ToDtype({tv_tensors.Image:torch.float32,"others":None}, scale=True),
    ])

def get_val_transforms(img_size):
    return T.Compose([
        T.Resize((img_size,img_size), antialias=True),
        T.ToDtype({tv_tensors.Image:torch.float32,"others":None}, scale=True),
    ])

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# 4. COORDINATE ATTENTION  (E2)
# ─────────────────────────────────────────────────────────────────────────────
class CoordinateAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None,1))
        self.pool_w = nn.AdaptiveAvgPool2d((1,None))
        self.conv1  = nn.Conv2d(ch, mid, 1, bias=False)
        self.bn1    = safe_gn(mid)
        self.act    = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, ch, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, ch, 1, bias=False)

    def forward(self, x):
        B,C,H,W = x.shape
        xh = self.pool_h(x)
        xw = self.pool_w(x).permute(0,1,3,2)
        y  = self.act(self.bn1(self.conv1(torch.cat([xh,xw],dim=2))))
        xh_out, xw_out = torch.split(y,[H,W],dim=2)
        xw_out = xw_out.permute(0,1,3,2)
        return x * torch.sigmoid(self.conv_h(xh_out)) * torch.sigmoid(self.conv_w(xw_out))


class AttentionFPNWrapper(nn.Module):
    def __init__(self, base_fpn, ch=256, reduction=32):
        super().__init__()
        self.fpn    = base_fpn
        self.attn_p2= CoordinateAttention(ch, reduction)
        self.attn_p3= CoordinateAttention(ch, reduction)

    def forward(self, x):
        out = self.fpn(x)
        if "0" in out: out["0"] = self.attn_p2(out["0"])
        if "1" in out: out["1"] = self.attn_p3(out["1"])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCALE-AWARE CONTEXT MODULE  (E3, E4, E5)
# ─────────────────────────────────────────────────────────────────────────────
class DilatedContextAggregator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        c1, c2, c3 = 24, 24, 16
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, dilation=1, bias=False),
            safe_gn(c1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2, 3, padding=2, dilation=2, bias=False),
            safe_gn(c2),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3, 3, padding=4, dilation=4, bias=False),
            safe_gn(c3),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(c1+c2+c3, out_channels, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.branch1(x),self.branch2(x),self.branch3(x)], dim=1))


class ScaleAwareContextModule(nn.Module):
    def __init__(self, in_channels: int = 256, n_scale_bins: int = 4, context_channels: int = 64):
        super().__init__()
        self.context_agg  = DilatedContextAggregator(in_channels, context_channels)
        self.scale_pred   = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_channels, n_scale_bins),
        )
        self.scale_embed  = nn.Linear(n_scale_bins, in_channels, bias=False)
        self.gate_proj    = nn.Conv2d(context_channels, in_channels, 1, bias=False)
        self.gate_sig     = nn.Sigmoid()
        self.output_proj  = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            safe_gn(in_channels),
        )

    def forward(self, features):
        identity = features
        B,C,H,W  = features.shape
        sw  = F.softmax(self.scale_pred(features), dim=1)
        emb = self.scale_embed(sw).view(B,C,1,1).expand_as(features)
        ctx = self.context_agg(features)
        gate= self.gate_sig(self.gate_proj(ctx))
        return self.output_proj(identity + emb * gate) + identity


class ScaleAwareRPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int,
                 n_scale_bins: int = 4, context_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            safe_gn(in_channels),
            nn.ReLU(inplace=True),
        )
        self.sacm       = ScaleAwareContextModule(in_channels, n_scale_bins, context_channels)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for l in [self.cls_logits, self.bbox_pred]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.zeros_(l.bias)

    def forward(self, features: List[torch.Tensor]):
        cls_out, box_out = [], []
        for f in features:
            t = self.sacm(self.conv(f))
            cls_out.append(self.cls_logits(t))
            box_out.append(self.bbox_pred(t))
        return cls_out, box_out


# ─────────────────────────────────────────────────────────────────────────────
# 6. E5 HIGH-RESOLUTION P1 BACKBONE
# ─────────────────────────────────────────────────────────────────────────────
class HighResP1Backbone(nn.Module):
    def __init__(self, base_backbone: nn.Module):
        super().__init__()
        self.body = base_backbone.body
        self.fpn  = base_backbone.fpn
        self.stem_lateral = nn.Conv2d(64, 256, 1, bias=False)
        self.p1_smooth    = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            safe_gn(256),
            nn.ReLU(inplace=True),
            CoordinateAttention(256, reduction=32),
        )
        self.out_channels = 256

    def forward(self, x):
        stem = self.body.relu(self.body.bn1(self.body.conv1(x)))
        c_feats = self.body(x)
        fpn_out = self.fpn(c_feats)
        p2 = fpn_out["0"]
        p2_up = F.interpolate(p2, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.p1_smooth(self.stem_lateral(stem) + p2_up)

        return {
            "p1": p1,
            "0":  fpn_out["0"],
            "1":  fpn_out["1"],
            "2":  fpn_out["2"],
            "3":  fpn_out["3"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. P1-AWARE FASTER RCNN (P1 -> RoI only, P2-P5 -> RPN)
# ─────────────────────────────────────────────────────────────────────────────
class P1HRDetFasterRCNN(FasterRCNN):
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("targets must be provided in training mode")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        all_features = self.backbone(images.tensors)
        rpn_features = {k: v for k, v in all_features.items() if k != "p1"}

        proposals, proposal_losses = self.rpn(images, rpn_features, targets)
        detections, detector_losses = self.roi_heads(
            all_features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        return detections


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def build_model(experiment: str = "baseline",
                num_classes: int = VISDRONE_NUM_CLASSES,
                pretrained: bool = True,
                img_size: int = 1280) -> Tuple[nn.Module, str]:

    if experiment in ("e5", "e5_p1_hrdet"):
        anchor_sizes = (
            (12,),     # P2 stride-4: 12px
            (24,),     # P3 stride-8: 24px
            (48,),     # P4 stride-16: 48px
            (96,),     # P5 stride-32: 96px
        )
        aspect_ratios = ((0.5, 1.0, 2.0),) * 4

        base_bb = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights="IMAGENET1K_V2" if pretrained else None,
            trainable_layers=3,
            returned_layers=[1,2,3,4],
        )
        backbone = HighResP1Backbone(base_bb)

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["p1", "0", "1", "2", "3"],
            output_size=7, sampling_ratio=2)

        model = P1HRDetFasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios),
            box_roi_pool=roi_pooler,
            box_nms_thresh=0.5,
            box_score_thresh=0.05,
            rpn_pre_nms_top_n_train=2000,
            rpn_post_nms_top_n_train=1000,
            rpn_pre_nms_top_n_test=3000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            min_size=img_size, max_size=img_size,
            image_mean=[0.485,0.456,0.406],
            image_std=[0.229,0.224,0.225],
        )
        na = model.rpn.head.cls_logits.out_channels
        model.rpn.head = ScaleAwareRPNHead(256, na, n_scale_bins=4, context_channels=64)
        print("[INFO] E5 (P1-HRDet): P1 High-Res RoI Align + P2-P5 Anchors (12-96px) + SACM (Ultra-Low VRAM <1.5GB)")
        return model, "E5_p1_hrdet"

    # E1-E4
    anchor_sizes  = ((16,),(32,),(64,),(128,),(256,))
    aspect_ratios = ((0.5,1.0,2.0,3.0),) * 5
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights="IMAGENET1K_V2" if pretrained else None,
        trainable_layers=3, returned_layers=[1,2,3,4])

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios),
        box_nms_thresh=0.5, box_score_thresh=0.05,
        rpn_pre_nms_top_n_train=4000, rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=6000,  rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        min_size=img_size, max_size=img_size,
        image_mean=[0.485,0.456,0.406], image_std=[0.229,0.224,0.225],
    )

    if pretrained:
        try:
            ref = FasterRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False)
            own = model.state_dict()
            matched = {k:v for k,v in ref.items() if k in own and own[k].shape == v.shape}
            own.update(matched); model.load_state_dict(own)
            print(f"[INFO] Loaded {len(matched)} layers from COCO V1 pretrained weights.")
        except Exception as e:
            print(f"[WARNING] COCO preload failed: {e}")

    if experiment in ("attention","final_model"):
        model.backbone.fpn = AttentionFPNWrapper(model.backbone.fpn, 256, 32)
        print("[INFO] Coordinate Attention added to FPN P2/P3.")

    if experiment in ("second_module","final_model"):
        na = model.rpn.head.cls_logits.out_channels
        model.rpn.head = ScaleAwareRPNHead(256, na, n_scale_bins=4, context_channels=64)
        print("[INFO] RPN head replaced with SACM.")

    names = {"baseline":"E1_baseline","attention":"E2_attention",
             "second_module":"E3_sacm","final_model":"E4_full"}
    return model, names.get(experiment, experiment)


# ─────────────────────────────────────────────────────────────────────────────
# 9. TRAINER
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpochMetrics:
    epoch: int
    loss_total: float; loss_rpn_cls: float; loss_rpn_box: float
    loss_roi_cls: float; loss_roi_box: float; lr: float
    map50: float=0.; map50_95: float=0.; ap_small: float=0.
    ap_medium: float=0.; ap_large: float=0.
    precision: float=0.; recall: float=0.; f1: float=0.


def compute_coco_eval(preds: List[Dict], val_coco: COCO) -> Dict[str,float]:
    zeros = {k:0. for k in ["map50","map50_95","ap_small","ap_medium","ap_large","precision","recall","f1"]}
    if not preds: return zeros
    dt  = val_coco.loadRes(preds)
    ev  = COCOeval(val_coco, dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    pr, re = float(s[0]), float(s[6])
    return {"map50_95":float(s[0]),"map50":float(s[1]),
            "ap_small":float(s[3]),"ap_medium":float(s[4]),"ap_large":float(s[5]),
            "precision":pr,"recall":re,"f1":(2*pr*re/(pr+re)) if pr+re>0 else 0.}


class Trainer:
    def __init__(self, model, train_loader, val_loader, val_coco,
                 experiment_name, epochs=50, lr=0.005,
                 img_size=1280, freeze_epochs=5, max_hours=8.0, device="cuda"):
        self.dev = self.device = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.model = model.to(self.dev)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.val_coco     = val_coco
        self.exp_name     = experiment_name
        self.epochs       = epochs
        self.base_lr      = lr
        self.img_size     = img_size
        self.freeze_epochs= freeze_epochs
        self.max_hours    = max_hours
        self.start_time   = time.time()
        self.out          = Path("results") / experiment_name
        for d in ["checkpoints","logs","metrics","predictions"]:
            (self.out/d).mkdir(parents=True, exist_ok=True)

        backbone_params = list(self.model.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]

        if self.freeze_epochs > 0:
            for p in backbone_params:
                p.requires_grad = False
            self.optimizer = SGD([
                {"params": head_params, "lr": lr},
                {"params": backbone_params, "lr": 0.0}
            ], momentum=0.9, weight_decay=1e-4, nesterov=True)
        else:
            self.optimizer = SGD([
                {"params": head_params, "lr": lr},
                {"params": backbone_params, "lr": lr * 0.1}
            ], momentum=0.9, weight_decay=1e-4, nesterov=True)

        self.scaler    = GradScaler(enabled=(self.dev.type=="cuda"))
        self.best_map  = -1.
        self.history: List[EpochMetrics] = []

        self.start_epoch = 1
        best_ckpt = self.out / "checkpoints" / "best.pth"
        last_ckpt = self.out / "checkpoints" / "last.pth"

        ckpt_to_load = None
        if best_ckpt.exists():
            try:
                b_data = torch.load(best_ckpt, map_location=self.dev)
                if isinstance(b_data, dict) and "metrics" in b_data:
                    if b_data["metrics"].get("map50", 0.0) > 0.01:
                        ckpt_to_load = best_ckpt
            except Exception:
                pass

        if ckpt_to_load is None and last_ckpt.exists():
            ckpt_to_load = last_ckpt

        if ckpt_to_load is not None:
            try:
                ckpt = torch.load(ckpt_to_load, map_location=self.dev)
                self.model.load_state_dict(ckpt.get("model", ckpt))
                self.start_epoch = ckpt.get("epoch", 0) + 1
                if "metrics" in ckpt and "map50" in ckpt["metrics"]:
                    self.best_map = ckpt["metrics"]["map50"]
                print(f"[INFO] RESUMED! Loaded {ckpt_to_load} -> Starting from Epoch {self.start_epoch} (mAP@50={self.best_map:.4f})")
            except Exception as e:
                print(f"[WARNING] Could not resume from {ckpt_to_load}: {e}")

    def _lr(self, epoch):
        wu = 3
        if epoch <= wu: return self.base_lr*(0.1+0.9*epoch/wu)
        prog = (epoch-wu)/max(1,self.epochs-wu)
        return 1e-5 + 0.5*(self.base_lr-1e-5)*(1+math.cos(math.pi*prog))

    def train_epoch(self, epoch):
        self.model.train()
        lr = self._lr(epoch)

        if epoch == self.freeze_epochs + 1:
            for p in self.model.backbone.parameters():
                p.requires_grad = True
            print(f"[E{epoch:03d}] Backbone unfrozen (requires_grad=True).")

        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = (lr * 0.1) if epoch > self.freeze_epochs else 0.0
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        running = {}
        for step,(images,targets) in enumerate(self.train_loader,1):
            images  = [im.to(self.dev,non_blocking=True) for im in images]
            targets = [{k:v.to(self.dev,non_blocking=True) for k,v in t.items()} for t in targets]
            try:
                with autocast(enabled=(self.dev.type=="cuda")):
                    ld = self.model(images,targets)
                    loss = sum(ld.values())

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss encountered at E{epoch:03d}/S{step:04d}. Skipping step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer); self.scaler.update()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARNING] CUDA OOM at E{epoch:03d}/S{step:04d}. Clearing cache & skipping step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            for k,v in ld.items(): running[k] = running.get(k,0.)+float(v.detach())
            running["total"] = running.get("total",0.)+float(loss.detach())
            if step%20==0 or step==len(self.train_loader):
                print(f"[{self.exp_name}][E{epoch:03d}/S{step:04d}] lr={lr:.5f} loss={running['total']/step:.4f}")
            if step%50==0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {k:v/max(1,len(self.train_loader)) for k,v in running.items()}

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        preds = []
        for images,targets in self.val_loader:
            images = [im.to(self.dev) for im in images]
            outs   = self.model(images)
            for t,o in zip(targets,outs):
                img_id = int(t["image_id"][0].item())
                meta   = self.val_coco.loadImgs([img_id])[0]
                sx     = meta.get("width",self.img_size)/self.img_size
                sy     = meta.get("height",self.img_size)/self.img_size
                for box,lbl,scr in zip(o["boxes"].cpu(),o["labels"].cpu(),o["scores"].cpu()):
                    x1,y1,x2,y2 = box.tolist()
                    preds.append({"image_id":img_id,"category_id":int(lbl),
                                  "bbox":[x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy],
                                  "score":float(scr)})
        return compute_coco_eval(preds, self.val_coco)

    @torch.inference_mode()
    def save_qualitative_predictions(self, num_samples: int = 15, conf_thresh: float = 0.25):
        self.model.eval()
        pred_dir = self.out / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        for images, targets in self.val_loader:
            images_gpu = [im.to(self.dev) for im in images]
            outs = self.model(images_gpu)
            for img_tensor, target, out in zip(images, targets, outs):
                if saved_count >= num_samples:
                    break
                img_id = int(target["image_id"][0].item())
                meta = self.val_coco.loadImgs([img_id])[0]
                file_name = meta.get("file_name", f"img_{img_id}.jpg")

                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                keep = out["scores"].cpu() >= conf_thresh
                p_boxes = out["boxes"].cpu()[keep].numpy()
                p_labels = out["labels"].cpu()[keep].numpy()
                p_scores = out["scores"].cpu()[keep].numpy()

                for box, lbl, scr in zip(p_boxes, p_labels, p_scores):
                    x1, y1, x2, y2 = map(int, box)
                    c_name = VISDRONE_CLASS_NAMES[lbl] if lbl < len(VISDRONE_CLASS_NAMES) else f"cls{lbl}"
                    color = CLASS_COLORS.get(lbl, (0, 255, 0))
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_bgr, f"{c_name} {scr:.2f}", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                out_path = pred_dir / f"val_{saved_count+1:02d}_{Path(file_name).stem}_det.jpg"
                cv2.imwrite(str(out_path), img_bgr)
                saved_count += 1
            if saved_count >= num_samples:
                break
        print(f"[INFO] Saved {saved_count} qualitative prediction images to {pred_dir}")

    def run(self):
        print(f"\n{'='*70}\n{self.exp_name}  |  {self.epochs} epochs  |  Max Hours: {self.max_hours}h  |  {self.dev}\n{'='*70}")
        if self.freeze_epochs > 0:
            for p in self.model.backbone.parameters(): p.requires_grad = False

        for epoch in range(self.start_epoch, self.epochs+1):
            tl = self.train_epoch(epoch)
            vm = self.validate()
            m  = EpochMetrics(epoch=epoch,
                loss_total=tl.get("total",0.),loss_rpn_cls=tl.get("loss_objectness",0.),
                loss_rpn_box=tl.get("loss_rpn_box_reg",0.),
                loss_roi_cls=tl.get("loss_classifier",0.),
                loss_roi_box=tl.get("loss_box_reg",0.),
                lr=self._lr(epoch), map50=vm["map50"], map50_95=vm["map50_95"],
                ap_small=vm["ap_small"], ap_medium=vm["ap_medium"], ap_large=vm["ap_large"],
                precision=vm["precision"], recall=vm["recall"], f1=vm["f1"])
            self.history.append(m)
            elapsed_h = (time.time() - self.start_time) / 3600.0
            print(f"[{self.exp_name}][E{epoch:03d}] mAP50={m.map50:.4f} "
                  f"mAP50:95={m.map50_95:.4f} AP_s={m.ap_small:.4f} F1={m.f1:.4f} (Elapsed: {elapsed_h:.2f}h / {self.max_hours:.1f}h)")

            state={"epoch":epoch,"model":self.model.state_dict(),"metrics":vm}
            torch.save(state, self.out/"checkpoints"/"last.pth")
            if m.map50 > self.best_map:
                self.best_map = m.map50
                torch.save(state, self.out/"checkpoints"/"best.pth")
                print(f"  ★ Best mAP@50: {self.best_map:.4f}")

            self.save_qualitative_predictions(num_samples=15)

            if elapsed_h >= self.max_hours:
                print(f"\n[INFO] Reached max execution time limit ({self.max_hours:.1f} hours). Stopping gracefully.")
                break

        if self.history:
            with open(self.out/"metrics"/"metrics.json","w") as f: json.dump(vm,f,indent=2)
            with open(self.out/"logs"/"training_log.csv","w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(self.history[0].__dict__.keys()))
                w.writeheader()
                for r in self.history: w.writerow(r.__dict__)


# ─────────────────────────────────────────────────────────────────────────────
# 10. COMPARISON SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def generate_comparison_summary():
    root  = Path("results")
    exps  = ["E1_baseline","E2_attention","E3_sacm","E4_full","E5_p1_hrdet"]
    rows  = []
    for e in exps:
        mf = root/e/"metrics"/"metrics.json"
        if mf.exists():
            d = json.loads(mf.read_text())
            rows.append({"Experiment":e,
                "mAP@50":d.get("map50",0.),"mAP@50:95":d.get("map50_95",0.),
                "AP_small":d.get("ap_small",0.),"AP_medium":d.get("ap_medium",0.),
                "AP_large":d.get("ap_large",0.),"Precision":d.get("precision",0.),
                "Recall":d.get("recall",0.),"F1":d.get("f1",0.)})
    if not rows: print("[WARNING] No completed metrics found."); return
    out = root/"experiment_comparison.csv"
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n{'='*75}\nABLATION RESULTS (E1-E5)\n{'='*75}")
    for r in rows:
        print(f"{r['Experiment']:<15} | mAP50={r['mAP@50']:.4f} | "
              f"mAP50:95={r['mAP@50:95']:.4f} | AP_small={r['AP_small']:.4f} | F1={r['F1']:.4f}")
    print(f"{'='*75}\nSaved → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 11. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def test_image(image_path, checkpoint_path, experiment="e5", conf=0.35, device="cuda"):
    dev   = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
    model,_ = build_model(experiment, pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(ckpt.get("model",ckpt))
    model.to(dev).eval()

    bgr  = cv2.imread(image_path)
    if bgr is None: print(f"[ERROR] Cannot read {image_path}"); return
    oh,ow = bgr.shape[:2]
    t = torch.from_numpy(cv2.cvtColor(cv2.resize(bgr,(1280,1280)),cv2.COLOR_BGR2RGB)
                         ).permute(2,0,1).float().div(255.).to(dev)
    t0 = time.time()
    with torch.no_grad(): out = model([t])[0]
    ms = (time.time()-t0)*1000

    keep   = out["scores"].cpu() >= conf
    boxes  = out["boxes"].cpu()[keep].numpy()
    labels = out["labels"].cpu()[keep].numpy()
    scores = out["scores"].cpu()[keep].numpy()
    sx,sy  = ow/1280., oh/1280.
    print(f"\n[{Path(image_path).name}]  {ms:.1f}ms  |  {len(boxes)} detections (conf≥{conf})")
    for b,l,s in zip(boxes,labels,scores):
        x1,y1,x2,y2 = int(b[0]*sx),int(b[1]*sy),int(b[2]*sx),int(b[3]*sy)
        name  = VISDRONE_CLASS_NAMES[l] if l<len(VISDRONE_CLASS_NAMES) else f"cls{l}"
        color = CLASS_COLORS.get(l,(0,255,0))
        cv2.rectangle(bgr,(x1,y1),(x2,y2),color,2)
        cv2.putText(bgr,f"{name} {s:.2f}",(x1,max(15,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        print(f"  {name:<15} {s:.3f}  [{x1},{y1},{x2},{y2}]  {x2-x1}×{y2-y1}px")
    op = Path("results/predictions") / f"{Path(image_path).stem}_det.jpg"
    op.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(op), bgr)
    print(f"Saved → {op}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       choices=["debug","research","all","test_image"], default="research")
    p.add_argument("--experiment", choices=["baseline","attention","second_module","final_model","e5"], default="e5")
    p.add_argument("--dataset_root", default="Dataset")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--image",      default="")
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--epochs",     type=int,   default=0)
    p.add_argument("--batch_size", type=int,   default=0)
    p.add_argument("--max_hours",  type=float, default=8.0)
    args = p.parse_args()

    set_seed(42)

    if args.mode == "test_image":
        if not args.checkpoint or not args.image:
            print("[ERROR] --mode test_image needs --checkpoint and --image"); sys.exit(1)
        test_image(args.image, args.checkpoint, args.experiment, args.confidence, args.device)
        return

    ds = Path(args.dataset_root)
    train_root = next((x for x in [
        ds/"VisDrone2019-DET-train"/"VisDrone2019-DET-train",
        ds/"VisDrone2019-DET-train", ds] if x.exists()), ds)
    val_root   = next((x for x in [ds/"VisDrone2019-DET-val", ds] if x.exists()), ds)

    if args.mode == "debug":
        epochs=args.epochs or 2; bs=args.batch_size or 2
        img_size=640; frac=0.05
    else:
        epochs=args.epochs or 50; bs=args.batch_size or 4
        img_size=1280; frac=1.0

    print(f"[INFO] Dataset: img_size={img_size}  batch={bs}  epochs={epochs}  max_hours={args.max_hours}h")
    train_ds = VisDroneDataset(train_root,"train",get_train_transforms(img_size),min_box_area=1.0)
    val_ds   = VisDroneDataset(val_root,  "val",  get_val_transforms(img_size),  min_box_area=1.0)
    if frac < 1.0:
        n = max(1,int(len(train_ds)*frac))
        train_ds = Subset(train_ds, list(range(n)))
        print(f"[DEBUG] Subset: {n} training images")

    tl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    vc = val_ds.coco

    exps = ["baseline","attention","second_module","final_model","e5"] if args.mode=="all" else [args.experiment]
    for exp in exps:
        model, name = build_model(exp, pretrained=True, img_size=img_size)
        Trainer(model, tl, vl, vc, name, epochs=epochs, lr=0.005,
                img_size=img_size, freeze_epochs=3 if args.mode=="debug" else 5,
                max_hours=args.max_hours, device=args.device).run()

    generate_comparison_summary()


if __name__ == "__main__":
    main()
