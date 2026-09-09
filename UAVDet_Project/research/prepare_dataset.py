"""
research/prepare_dataset.py
============================
VisDrone-DET TXT → COCO JSON converter and dataset validator.

VisDrone annotation format (per line):
  x, y, w, h, score, category_id, truncation, occlusion

  - score=0  → ignored region (excluded)
  - score=1  → valid annotation
  - category_id: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car,
                 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor

Usage:
  python research/prepare_dataset.py              # convert + validate
  python research/prepare_dataset.py --validate   # validate only (if JSON already exists)
  python research/prepare_dataset.py --test       # run dataset loader test

Output:
  Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/annotations/instances_train.json
  Dataset/VisDrone2019-DET-val/annotations/instances_val.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# VisDrone metadata
# ──────────────────────────────────────────────────────────────────────────────

VISDRONE_CATEGORIES = [
    {"id": 1,  "name": "pedestrian",      "supercategory": "person"},
    {"id": 2,  "name": "people",          "supercategory": "person"},
    {"id": 3,  "name": "bicycle",         "supercategory": "vehicle"},
    {"id": 4,  "name": "car",             "supercategory": "vehicle"},
    {"id": 5,  "name": "van",             "supercategory": "vehicle"},
    {"id": 6,  "name": "truck",           "supercategory": "vehicle"},
    {"id": 7,  "name": "tricycle",        "supercategory": "vehicle"},
    {"id": 8,  "name": "awning-tricycle", "supercategory": "vehicle"},
    {"id": 9,  "name": "bus",             "supercategory": "vehicle"},
    {"id": 10, "name": "motor",           "supercategory": "vehicle"},
]

# Canonical dataset split roots relative to project root
SPLIT_DIRS = {
    "train": Path("Dataset/VisDrone2019-DET-train/VisDrone2019-DET-train"),
    "val":   Path("Dataset/VisDrone2019-DET-val"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Converter
# ──────────────────────────────────────────────────────────────────────────────

def _img_id_from_name(name: str, idx: int) -> int:
    """Extract numeric ID from filename or use index as fallback."""
    nums = re.findall(r"\d+", Path(name).stem)
    if nums:
        # Use last long number in filename as image_id
        return int(max(nums, key=len))
    return idx + 1


def convert_split(split_dir: Path, split: str, min_box_side: int = 2) -> Dict:
    """Convert all TXT annotation files for one split to COCO JSON dict."""
    images_dir      = split_dir / "images"
    annotations_dir = split_dir / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations/ not found: {annotations_dir}")

    img_files = sorted(images_dir.glob("*.jpg"))
    if not img_files:
        img_files = sorted(images_dir.glob("*.JPG"))
    if not img_files:
        img_files = sorted(images_dir.glob("*.png"))

    print(f"  [{split}] Found {len(img_files)} images in {images_dir}")

    # Read image sizes
    try:
        import cv2
        def _img_size(path):
            img = cv2.imread(str(path))
            if img is None:
                return 1920, 1080  # VisDrone default
            return img.shape[1], img.shape[0]  # width, height
    except ImportError:
        try:
            from PIL import Image as PIL_Image
            def _img_size(path):
                with PIL_Image.open(str(path)) as im:
                    return im.size  # (W, H)
        except ImportError:
            def _img_size(path):
                return 1920, 1080  # fallback

    coco_images   = []
    coco_anns     = []
    ann_id        = 1
    skipped_boxes = 0
    total_boxes   = 0

    for img_idx, img_path in enumerate(img_files):
        stem    = img_path.stem
        img_id  = img_idx + 1   # sequential 1-indexed

        # Get image dimensions
        try:
            W, H = _img_size(img_path)
        except Exception:
            W, H = 1920, 1080

        coco_images.append({
            "id":        img_id,
            "file_name": img_path.name,
            "width":     W,
            "height":    H,
        })

        # Read corresponding annotation TXT
        txt_path = annotations_dir / f"{stem}.txt"
        if not txt_path.exists():
            continue

        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            try:
                x, y, w, h   = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                score        = int(parts[4])
                category_id  = int(parts[5])
            except ValueError:
                continue

            total_boxes += 1

            # Skip ignored regions (score=0) and invalid categories
            if score == 0 or category_id == 0 or category_id > 10:
                skipped_boxes += 1
                continue

            # Clamp and validate
            x  = max(0.0, x)
            y  = max(0.0, y)
            w  = min(w, W - x)
            h  = min(h, H - y)

            if w < min_box_side or h < min_box_side:
                skipped_boxes += 1
                continue

            area = w * h

            coco_anns.append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  category_id,
                "bbox":         [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                "area":         round(area, 1),
                "iscrowd":      0,
                "truncation":   int(parts[6]) if len(parts) > 6 else 0,
                "occlusion":    int(parts[7]) if len(parts) > 7 else 0,
            })
            ann_id += 1

    valid_boxes = total_boxes - skipped_boxes
    print(f"  [{split}] {len(coco_images)} images  |  {valid_boxes} valid annotations  "
          f"({skipped_boxes} skipped: ignored/too small)")

    return {
        "info": {
            "description":  "VisDrone-DET 2019",
            "version":      "1.0",
            "split":        split,
            "year":         2019,
            "contributor":  "AISKYEYE Lab",
        },
        "licenses":    [],
        "categories":  VISDRONE_CATEGORIES,
        "images":      coco_images,
        "annotations": coco_anns,
    }


def convert_all_splits(project_root: Path, splits: Optional[List[str]] = None) -> None:
    """Convert VisDrone TXT annotations to COCO JSON for specified splits."""
    if splits is None:
        splits = ["train", "val"]

    for split in splits:
        rel_dir = SPLIT_DIRS[split]
        split_dir = project_root / rel_dir
        if not split_dir.exists():
            print(f"  [WARNING] Split directory not found: {split_dir}")
            continue

        ann_dir = split_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        out_path = ann_dir / f"instances_{split}.json"

        if out_path.exists():
            existing = json.loads(out_path.read_text())
            n_imgs = len(existing.get("images", []))
            n_anns = len(existing.get("annotations", []))
            print(f"  [{split}] JSON already exists: {n_imgs} images, {n_anns} annotations")
            ans = input(f"  Overwrite {out_path.name}? [y/N]: ").strip().lower()
            if ans != "y":
                print(f"  [{split}] Keeping existing JSON.")
                continue

        print(f"\n[CONVERT] {split} split -> {out_path.name}")
        coco_dict = convert_split(split_dir, split)

        out_path.write_text(json.dumps(coco_dict, indent=2, ensure_ascii=False))
        print(f"  [{split}] Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────

def validate_coco_json(project_root: Path) -> bool:
    """Validate that COCO JSON files exist and have correct structure."""
    from pycocotools.coco import COCO

    all_ok = True
    for split, rel_dir in SPLIT_DIRS.items():
        json_path = project_root / rel_dir / "annotations" / f"instances_{split}.json"
        if not json_path.exists():
            print(f"  [FAIL] {split}: JSON not found at {json_path}")
            all_ok = False
            continue

        try:
            coco = COCO(str(json_path))
        except Exception as e:
            print(f"  [FAIL] {split}: COCO load error: {e}")
            all_ok = False
            continue

        n_imgs = len(coco.getImgIds())
        n_anns = len(coco.getAnnIds())
        cats   = sorted([c["name"] for c in coco.loadCats(coco.getCatIds())])
        print(f"  [OK]   {split}: {n_imgs} images, {n_anns} annotations, "
              f"{len(cats)} categories: {cats}")

        # Validate bbox format
        ann_ids = coco.getAnnIds()[:50]
        for ann in coco.loadAnns(ann_ids):
            b = ann["bbox"]
            assert len(b) == 4, f"bbox must have 4 elements"
            assert b[2] > 0 and b[3] > 0, f"w and h must be >0: {b}"

        print(f"         (first 50 annotations: bboxes all valid)")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader test (uses real VisDrone data)
# ──────────────────────────────────────────────────────────────────────────────

def test_dataset_loader(project_root: Path, n_samples: int = 5) -> bool:
    """Test the VisDroneDataset loader with real data."""
    print("\n[TEST] VisDroneDataset loader test...")
    try:
        from research.datasets.visdrone import VisDroneDataset, get_train_transforms, get_val_transforms
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False

    import torch

    PASS = FAIL = 0

    def _ok(msg):  nonlocal PASS; print(f"  PASS  {msg}"); PASS += 1
    def _fail(msg): nonlocal FAIL; print(f"  FAIL  {msg}"); FAIL += 1

    for split in ["val", "train"]:
        rel_dir = SPLIT_DIRS[split]
        split_dir = project_root / rel_dir
        json_path = split_dir / "annotations" / f"instances_{split}.json"
        if not json_path.exists():
            _fail(f"{split}: JSON not found — run convert first")
            continue

        try:
            ds = VisDroneDataset(split_dir, split, transforms=None)
            _ok(f"{split}: dataset loaded  ({len(ds)} images)")
        except Exception as e:
            _fail(f"{split}: VisDroneDataset init failed: {e}")
            continue

        # Test without transforms
        try:
            img, target = ds[0]
            assert img.dtype == torch.float32,  f"image dtype {img.dtype}"
            assert img.shape[0] == 3,           f"image channels {img.shape}"
            assert img.min() >= 0.0,            f"image min < 0"
            assert img.max() <= 1.0,            f"image max > 1"
            _ok(f"{split}[0]: image  {tuple(img.shape)}  dtype={img.dtype}")
        except Exception as e:
            _fail(f"{split}[0]: image load failed: {e}"); continue

        try:
            boxes  = target["boxes"]
            labels = target["labels"]
            assert boxes.dtype == torch.float32,  f"boxes dtype {boxes.dtype}"
            assert labels.dtype == torch.int64,   f"labels dtype {labels.dtype}"
            assert boxes.shape[1] == 4,           f"boxes shape {boxes.shape}"
            assert (boxes[:, 0] < boxes[:, 2]).all(), "x1 < x2 violated"
            assert (boxes[:, 1] < boxes[:, 3]).all(), "y1 < y2 violated"
            assert not torch.isnan(boxes).any(),  "NaN in boxes"
            _ok(f"{split}[0]: boxes  {tuple(boxes.shape)}  labels  {tuple(labels.shape)}")
        except Exception as e:
            _fail(f"{split}[0]: target validation failed: {e}"); continue

    # Test with train transforms (the critical fix)
    try:
        rel_dir = SPLIT_DIRS["train"]
        split_dir = project_root / rel_dir
        json_path = split_dir / "annotations" / "instances_train.json"
        if json_path.exists():
            ds_aug = VisDroneDataset(split_dir, "train",
                                     transforms=get_train_transforms(img_size=640))
            import random
            for i in random.sample(range(min(len(ds_aug), 100)), min(n_samples, 5)):
                img, target = ds_aug[i]
                b = target["boxes"]
                assert b.dtype == torch.float32,      f"boxes dtype after aug: {b.dtype}"
                assert target["labels"].dtype == torch.int64, "labels not int64"
                if len(b) > 0:
                    assert (b[:, 2] > b[:, 0]).all(), "x2<=x1 after augment"
                    assert (b[:, 3] > b[:, 1]).all(), "y2<=y1 after augment"
                    assert not torch.isnan(b).any(),  "NaN in augmented boxes"
            _ok(f"train transforms: RandomIoUCrop + RandomZoomOut work correctly  "
                f"({n_samples} samples tested)")
        else:
            print(f"  [SKIP] train transforms test: JSON not found")
    except Exception as e:
        _fail(f"train transforms failed: {e}")
        import traceback; traceback.print_exc()

    print(f"\n[TEST DONE] {PASS} PASS  {FAIL} FAIL")
    return FAIL == 0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset summary
# ──────────────────────────────────────────────────────────────────────────────

def print_dataset_summary(project_root: Path) -> None:
    """Print dataset statistics from COCO JSON files."""
    from collections import Counter
    from pycocotools.coco import COCO

    print("\n" + "="*60)
    print("VISDRONE DATASET SUMMARY")
    print("="*60)
    for split, rel_dir in SPLIT_DIRS.items():
        json_path = project_root / rel_dir / "annotations" / f"instances_{split}.json"
        if not json_path.exists():
            print(f"\n[{split.upper()}] JSON not found")
            continue

        coco = COCO(str(json_path))
        ann_ids = coco.getAnnIds()
        anns    = coco.loadAnns(ann_ids)
        cats    = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        counter = Counter(a["category_id"] for a in anns)

        print(f"\n[{split.upper()}]  {len(coco.getImgIds())} images  |  "
              f"{len(ann_ids)} annotations")
        print(f"  {'Class':<20} {'Count':>8}  {'%':>6}")
        print(f"  {'-'*38}")
        for cat_id in sorted(cats):
            n = counter.get(cat_id, 0)
            pct = 100 * n / max(len(ann_ids), 1)
            print(f"  {cats[cat_id]:<20} {n:>8}  {pct:>5.1f}%")

        # Size distribution
        areas = [a["area"] for a in anns]
        small  = sum(1 for a in areas if a < 1024)
        medium = sum(1 for a in areas if 1024 <= a < 9216)
        large  = sum(1 for a in areas if a >= 9216)
        total  = len(areas)
        print(f"\n  Object size distribution (COCO definitions):")
        print(f"    small  (<32²px):    {small:>6}  ({100*small/total:.1f}%)")
        print(f"    medium (32²-96²px): {medium:>6}  ({100*medium/total:.1f}%)")
        print(f"    large  (>96²px):    {large:>6}  ({100*large/total:.1f}%)")
    print("="*60)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VisDrone TXT → COCO JSON converter and validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--convert",  action="store_true", default=True,
                        help="Convert TXT annotations to COCO JSON")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing COCO JSON files")
    parser.add_argument("--test",     action="store_true",
                        help="Run dataset loader test with real data")
    parser.add_argument("--summary",  action="store_true",
                        help="Print dataset statistics")
    parser.add_argument("--splits", nargs="+", default=["train","val"],
                        choices=["train","val"])
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Auto-confirm overwrite without prompting")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # Patch input() for --yes
    if args.yes:
        import builtins
        builtins.input = lambda _: "y"

    print(f"\nVisDrone Dataset Preparation")
    print(f"Project root: {project_root}")

    any_action = args.validate or args.test or args.summary or args.convert
    if not any_action:
        args.convert = True  # default action

    if args.convert:
        print(f"\n{'='*60}")
        print("STEP 1: Converting TXT -> COCO JSON")
        print("="*60)
        convert_all_splits(project_root, args.splits)

    if args.validate or args.convert:
        print(f"\n{'='*60}")
        print("STEP 2: Validating COCO JSON")
        print("="*60)
        ok = validate_coco_json(project_root)
        if not ok:
            print("\n[ERROR] Validation failed. Fix errors before training.")
            sys.exit(1)
        else:
            print("\n[OK] All COCO JSON files are valid.")

    if args.summary:
        print_dataset_summary(project_root)

    if args.test:
        ok = test_dataset_loader(project_root)
        if not ok:
            print("\n[ERROR] Dataset loader test failed.")
            sys.exit(1)
        else:
            print("\n[OK] Dataset loader test passed.")

    print("\nDataset preparation complete.")
    print("Next step: python research/run_experiment.py --experiment baseline --mode debug --device cuda")


if __name__ == "__main__":
    main()
