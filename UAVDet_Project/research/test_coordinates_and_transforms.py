"""
research/test_coordinates_and_transforms.py
===========================================
Comprehensive unit and integration test suite proving:
  1. Coordinate transformation & IoU correctness (Test 1: identical box IoU=1.0, Test 2: shifted box, Test 3: non-overlapping box)
  2. Training augmentation robust bounding box handling (no zero/negative/NaN boxes)
  3. Empty target handling (no crash, valid representation)
  4. Rescaling between model prediction space and original image space for COCOeval

Usage:
  python research/test_coordinates_and_transforms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.tv_tensors as tv_tensors
from torchvision.transforms import v2 as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.datasets.visdrone import (
    VisDroneDataset,
    get_train_transforms,
    get_val_transforms,
    build_dataloaders,
)
from research.evaluation.metrics import _box_iou, compute_coco_metrics
from research.pipeline import _iou_xyxy


def test_iou_math():
    print("\n[TEST 1] Testing IoU Mathematical Precision...")
    # Test 1A: Exact matching box -> IoU = 1.0
    b1 = [10.0, 20.0, 50.0, 80.0]
    b2 = [10.0, 20.0, 50.0, 80.0]
    iou_exact = _box_iou(b1, b2)
    assert abs(iou_exact - 1.0) < 1e-5, f"Expected 1.0, got {iou_exact}"
    print(f"  PASS: Exact box match IoU = {iou_exact:.4f} (Expected 1.0000)")

    # Test 1B: 50% horizontal shift
    b3 = [10.0, 20.0, 50.0, 80.0]  # w=40, h=60, area=2400
    b4 = [30.0, 20.0, 70.0, 80.0]  # overlap w=20, h=60, inter=1200, union=3600 -> iou=1/3=0.3333
    iou_shift = _box_iou(b3, b4)
    expected_shift = 1200.0 / 3600.0
    assert abs(iou_shift - expected_shift) < 1e-4, f"Expected {expected_shift}, got {iou_shift}"
    print(f"  PASS: Shifted box overlap IoU = {iou_shift:.4f} (Expected {expected_shift:.4f})")

    # Test 1C: Completely non-overlapping boxes
    b5 = [10.0, 20.0, 50.0, 80.0]
    b6 = [100.0, 200.0, 150.0, 280.0]
    iou_none = _box_iou(b5, b6)
    assert iou_none < 1e-5, f"Expected 0.0, got {iou_none}"
    print(f"  PASS: Disjoint box overlap IoU = {iou_none:.4f} (Expected 0.0000)")


def test_augmentation_invariants():
    print("\n[TEST 2] Testing Augmentation Pipeline Invariants (RandomIoUCrop & RandomZoomOut)...")
    val_root = PROJECT_ROOT / "Dataset/VisDrone2019-DET-val"
    train_root = PROJECT_ROOT / "Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train"

    train_ds = VisDroneDataset(
        root=train_root,
        split="train",
        transforms=get_train_transforms(img_size=640),
    )

    print(f"Testing 100 randomly sampled augmented samples from {len(train_ds)} train images...")
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(train_ds)), min(100, len(train_ds)))

    for idx in sample_indices:
        img, target = train_ds[idx]
        boxes = target["boxes"]
        labels = target["labels"]

        assert img.shape == (3, 640, 640), f"Invalid image tensor shape: {img.shape}"
        assert img.dtype == torch.float32, f"Invalid image dtype: {img.dtype}"
        assert img.min() >= 0.0 and img.max() <= 1.0, f"Image values out of range [0, 1]"

        assert len(boxes) > 0, "Target boxes must not be empty"
        assert len(boxes) == len(labels), f"Boxes count {len(boxes)} != labels count {len(labels)}"
        assert boxes.shape[1] == 4, f"Boxes must be [N, 4]"

        # Check positive width and height
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        assert (x2 > x1).all(), f"Found box with x2 <= x1 in image index {idx}: {boxes}"
        assert (y2 > y1).all(), f"Found box with y2 <= y1 in image index {idx}: {boxes}"
        assert not torch.isnan(boxes).any(), f"Found NaN in boxes in image index {idx}"
        assert not torch.isinf(boxes).any(), f"Found Inf in boxes in image index {idx}"

    print("  PASS: All 100 augmented samples strictly satisfy x2 > x1 and y2 > y1 with valid dtypes.")


def test_empty_target_handling():
    print("\n[TEST 3] Testing Synthetic Empty Target Handling...")
    ds = VisDroneDataset.__new__(VisDroneDataset)
    ds.transforms = get_train_transforms(640)
    ds.min_box_area = 4.0

    # Simulate an image with zero boxes
    dummy_img = torch.randint(0, 256, (3, 500, 500), dtype=torch.uint8)
    dummy_target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([9999], dtype=torch.int64),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
    }

    out_img, out_target = ds._apply_transforms(dummy_img, dummy_target, 500, 500)
    boxes = out_target["boxes"]
    assert len(boxes) == 1, "Fallback dummy box should be generated"
    assert (boxes[0, 2] > boxes[0, 0]) and (boxes[0, 3] > boxes[0, 1]), "Dummy box must have positive w and h"
    assert out_target["labels"][0].item() == 0, "Dummy box must have label 0 (background)"
    print(f"  PASS: Empty target cleanly produced background fallback box: {boxes.tolist()} with label {out_target['labels'].tolist()}")


def test_coordinate_rescaling_with_coco():
    print("\n[TEST 4] Testing Coordinate Rescaling & COCOeval Roundtrip...")
    val_root = PROJECT_ROOT / "Dataset/VisDrone2019-DET-val"
    train_root = PROJECT_ROOT / "Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train"

    img_size = 640
    _, val_loader, val_coco = build_dataloaders(
        dataset_root=val_root,
        img_size=img_size,
        batch_size=4,
        num_workers=0,
        seed=42,
        train_root=train_root,
        val_root=val_root,
    )

    # Pick 5 images from val_loader, simulate predictions in 640x640 space, and rescale them to original space
    predictions_scaled = []
    total_gt = 0

    for i, (images, targets) in enumerate(val_loader):
        if i >= 3:
            break
        for target in targets:
            img_id = int(target["image_id"][0].item())
            meta = val_coco.loadImgs([img_id])[0]
            orig_w, orig_h = meta["width"], meta["height"]
            scale_x = orig_w / img_size
            scale_y = orig_h / img_size

            for b, l in zip(target["boxes"], target["labels"]):
                x1, y1, x2, y2 = b.tolist()
                predictions_scaled.append({
                    "image_id": img_id,
                    "category_id": int(l.item()),
                    "bbox": [
                        x1 * scale_x,
                        y1 * scale_y,
                        (x2 - x1) * scale_x,
                        (y2 - y1) * scale_y,
                    ],
                    "score": 1.0,
                })
                total_gt += 1

    print(f"Evaluated {len(predictions_scaled)} rescaled predictions across 12 images against COCO ground truth.")
    metrics = compute_coco_metrics(predictions_scaled, [], coco_gt_api=val_coco)
    print(f"  PASS: Rescaled coordinate evaluation produced mAP@50 = {metrics['map50']:.4f}, Recall = {metrics['recall']:.4f}")
    assert metrics["map50"] > 0.0, "mAP@50 must be strictly positive for ground-truth aligned predictions!"


def main():
    print("=" * 70)
    print("RUNNING COORDINATE, TRANSFORM, AND AUGMENTATION VALIDATION SUITE")
    print("=" * 70)

    test_iou_math()
    test_augmentation_invariants()
    test_empty_target_handling()
    test_coordinate_rescaling_with_coco()

    print("\n" + "=" * 70)
    print("ALL COORDINATE AND TRANSFORM TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
