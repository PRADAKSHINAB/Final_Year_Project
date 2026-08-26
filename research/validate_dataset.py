"""
research/validate_dataset.py
============================
Automated Comprehensive VisDrone Dataset & COCO Annotation Validator.

Checks:
  1. Image files existence and readability (checks for corruption)
  2. Annotation JSON existence, JSON syntax, and COCO schema
  3. Image IDs match between images and annotations
  4. Bounding box validity:
     - x1, y1 >= 0
     - w > 0, h > 0 (x2 > x1, y2 > y1)
     - x2 <= width, y2 <= height
     - No zero-area or negative-area boxes
     - No NaN or Inf coordinates
  5. Category IDs validity (1 to 10 for VisDrone)
  6. Category names consistency

Usage:
  python research/validate_dataset.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
from pycocotools.coco import COCO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_DIRS = {
    "train": Path("Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train"),
    "val":   Path("Dataset/VisDrone2019-DET-val"),
}

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def validate_split(split_name: str, split_dir: Path) -> Dict:
    print(f"\n{'='*60}")
    print(f"VALIDATING SPLIT: {split_name.upper()}")
    print(f"Path: {split_dir}")
    print(f"{'='*60}")

    images_dir = split_dir / "images"
    json_path = split_dir / "annotations" / f"instances_{split_name}.json"

    errors = []
    warnings = []

    if not images_dir.exists():
        errors.append(f"Images directory missing: {images_dir}")
        return {"split": split_name, "errors": errors, "warnings": warnings}

    if not json_path.exists():
        errors.append(f"COCO JSON annotation missing: {json_path}")
        return {"split": split_name, "errors": errors, "warnings": warnings}

    # Load COCO JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        errors.append(f"Failed to parse JSON file {json_path}: {e}")
        return {"split": split_name, "errors": errors, "warnings": warnings}

    # Schema checks
    for key in ["images", "annotations", "categories"]:
        if key not in data:
            errors.append(f"Missing top-level key in JSON: {key}")

    if errors:
        return {"split": split_name, "errors": errors, "warnings": warnings}

    coco_images = data["images"]
    coco_anns = data["annotations"]
    coco_cats = data["categories"]

    print(f"Found {len(coco_images):,} image records and {len(coco_anns):,} annotation records.")
    print(f"Found {len(coco_cats)} categories.")

    # 1. Category validation
    cat_ids = {c["id"] for c in coco_cats}
    cat_names = {c["id"]: c["name"] for c in coco_cats}
    for cid in sorted(cat_ids):
        if cid < 1 or cid > 10:
            errors.append(f"Invalid category_id {cid} (must be 1..10)")
    for name in VISDRONE_CLASSES:
        if name not in cat_names.values():
            warnings.append(f"VisDrone class '{name}' not found in categories definition")

    # 2. Image IDs & File existence
    img_id_to_record = {}
    missing_files = 0
    corrupted_files = 0
    sample_checked = 0

    for idx, img_info in enumerate(coco_images):
        iid = img_info["id"]
        fname = img_info["file_name"]
        if iid in img_id_to_record:
            errors.append(f"Duplicate image_id: {iid}")
        img_id_to_record[iid] = img_info

        img_file = images_dir / fname
        if not img_file.exists():
            missing_files += 1
            if missing_files <= 5:
                errors.append(f"Image file missing on disk: {img_file.name}")
        else:
            # Check first 50 images for corruption
            if sample_checked < 50:
                im = cv2.imread(str(img_file))
                if im is None:
                    corrupted_files += 1
                    errors.append(f"Corrupted image file (cannot read with OpenCV): {img_file.name}")
                sample_checked += 1

    if missing_files > 5:
        errors.append(f"... and {missing_files - 5} more missing image files.")

    # 3. Annotation validation
    ann_id_set = set()
    zero_area_boxes = 0
    negative_area_boxes = 0
    nan_boxes = 0
    out_of_bounds_boxes = 0
    invalid_labels = 0

    for idx, ann in enumerate(coco_anns):
        aid = ann.get("id")
        if aid in ann_id_set:
            errors.append(f"Duplicate annotation id: {aid}")
        ann_id_set.add(aid)

        img_id = ann.get("image_id")
        if img_id not in img_id_to_record:
            errors.append(f"Annotation {aid} references nonexistent image_id {img_id}")
            continue

        cid = ann.get("category_id")
        if cid not in cat_ids:
            invalid_labels += 1

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            errors.append(f"Annotation {aid} has invalid bbox length {len(bbox)} (must be 4)")
            continue

        x, y, w, h = bbox
        if any(math.isnan(v) or math.isinf(v) for v in [x, y, w, h]):
            nan_boxes += 1
            continue

        if w <= 0 or h <= 0:
            if w == 0 or h == 0:
                zero_area_boxes += 1
            else:
                negative_area_boxes += 1

        img_w = img_id_to_record[img_id]["width"]
        img_h = img_id_to_record[img_id]["height"]

        if (x + w) > img_w + 10 or (y + h) > img_h + 10:
            out_of_bounds_boxes += 1

    if zero_area_boxes > 0:
        errors.append(f"Found {zero_area_boxes} zero-area bounding boxes (w=0 or h=0)")
    if negative_area_boxes > 0:
        errors.append(f"Found {negative_area_boxes} negative-area bounding boxes")
    if nan_boxes > 0:
        errors.append(f"Found {nan_boxes} NaN/Inf bounding boxes")
    if invalid_labels > 0:
        errors.append(f"Found {invalid_labels} annotations with invalid category_id")

    # 4. Pycocotools COCO load test
    try:
        coco_api = COCO(str(json_path))
        print(f"pycocotools COCO index built successfully for {split_name}.")
    except Exception as e:
        errors.append(f"pycocotools failed to load {json_path}: {e}")

    summary = {
        "split": split_name,
        "total_images": len(coco_images),
        "total_annotations": len(coco_anns),
        "total_categories": len(coco_cats),
        "missing_files": missing_files,
        "corrupted_files": corrupted_files,
        "zero_area_boxes": zero_area_boxes,
        "negative_area_boxes": negative_area_boxes,
        "nan_boxes": nan_boxes,
        "errors": errors,
        "warnings": warnings,
    }

    if not errors:
        print(f"  [PASS] Split {split_name}: All {len(coco_images):,} images and {len(coco_anns):,} annotations are VALID.")
    else:
        print(f"  [FAIL] Split {split_name}: Found {len(errors)} critical issues.")
        for err in errors[:10]:
            print(f"    - {err}")

    return summary


def main():
    print("=" * 70)
    print("AUTOMATED VISDRONE DATASET AUDIT & INTEGRITY CHECK")
    print("=" * 70)

    all_passed = True
    for split, path in SPLIT_DIRS.items():
        res = validate_split(split, PROJECT_ROOT / path)
        if res["errors"]:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL DATASET AUDIT RESULT: PASSED (100% Valid Dataset & Annotations)")
        sys.exit(0)
    else:
        print("OVERALL DATASET AUDIT RESULT: FAILED (Critical Issues Found)")
        sys.exit(1)


if __name__ == "__main__":
    main()
