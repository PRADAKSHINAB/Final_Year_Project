"""
research/check_dependencies.py
==============================
Dependency Checker for UAV Small-Object Detection System.

Checks all required python packages, imports, and optional utilities.

Usage:
  python research/check_dependencies.py
"""
from __future__ import annotations

import importlib
import sys

REQUIRED_PACKAGES = [
    ("torch", "PyTorch deep learning framework"),
    ("torchvision", "TorchVision models & transforms"),
    ("pycocotools", "COCO evaluation metrics (pycocotools)"),
    ("cv2", "OpenCV image processing (opencv-python)"),
    ("PIL", "Python Imaging Library (Pillow)"),
    ("matplotlib", "Plotting & publication figures"),
    ("numpy", "Numerical computing"),
    ("yaml", "YAML configuration parser (PyYAML)"),
]

OPTIONAL_PACKAGES = [
    ("fvcore", "FLOPs computation (fvcore)"),
]


def check():
    print("=" * 60)
    print("DEPENDENCY AUDIT & IMPORT CHECK")
    print("=" * 60)

    missing_required = []

    print("\n[REQUIRED DEPENDENCIES]")
    for mod_name, desc in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  PASS  {mod_name:<15} ({ver}) — {desc}")
        except ImportError:
            print(f"  FAIL  {mod_name:<15} NOT FOUND — {desc}")
            missing_required.append(mod_name)

    print("\n[OPTIONAL DEPENDENCIES]")
    for mod_name, desc in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  PASS  {mod_name:<15} ({ver}) — {desc}")
        except ImportError:
            print(f"  WARN  {mod_name:<15} NOT FOUND — {desc}")

    print("\n" + "=" * 60)
    if missing_required:
        print(f"RESULT: {len(missing_required)} REQUIRED DEPENDENCIES MISSING: {', '.join(missing_required)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("RESULT: ALL REQUIRED DEPENDENCIES ARE INSTALLED")
    print("=" * 60)


if __name__ == "__main__":
    check()
