"""
research/check_hpc.py
======================
HPC Environment & Hardware Validation Script for H200 GPU.

Checks:
  - Python version
  - PyTorch version
  - CUDA availability, version, device count, GPU name, total VRAM
  - torchvision version
  - OpenCV version
  - pycocotools availability
  - Basic GPU tensor operations & CUDA memory allocation
  - Model forward pass on GPU (with AMP if supported)
  - Optional batch size / VRAM stress test

Usage:
  python research/check_hpc.py
  python research/check_hpc.py --batch_test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_environment() -> bool:
    print("=" * 60)
    print("HPC HARDWARE & ENVIRONMENT VALIDATION")
    print("=" * 60)

    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"PyTorch Version     : {torch.__version__}")

    # torchvision
    try:
        import torchvision
        print(f"torchvision Version : {torchvision.__version__}")
    except ImportError:
        print("torchvision Version : NOT INSTALLED")

    # opencv
    try:
        import cv2
        print(f"OpenCV Version      : {cv2.__version__}")
    except ImportError:
        print("OpenCV Version      : NOT INSTALLED")

    # pycocotools
    try:
        import pycocotools
        print(f"pycocotools         : INSTALLED ({getattr(pycocotools, '__version__', 'OK')})")
    except ImportError:
        print("pycocotools         : NOT INSTALLED")

    # fvcore
    try:
        import fvcore
        print(f"fvcore              : INSTALLED ({getattr(fvcore, '__version__', 'OK')})")
    except ImportError:
        print("fvcore              : NOT INSTALLED (GFLOPs will use fallback/null)")

    # CUDA / GPU
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available      : {cuda_available}")

    if cuda_available:
        print(f"CUDA Version        : {torch.version.cuda}")
        print(f"cuDNN Version       : {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count           : {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  Device {i}           : {props.name} ({vram_gb:.2f} GB VRAM, SM {props.major}.{props.minor})")
        dev = torch.device("cuda:0")
    else:
        print("CUDA Device         : CPU Only")
        dev = torch.device("cpu")

    print("-" * 60)
    return cuda_available


def test_tensor_ops(device: torch.device) -> bool:
    print(f"\n[TEST 1] Tensor allocation & matrix multiplication on {device}...")
    try:
        t0 = time.perf_counter()
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        c = torch.matmul(a, b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  PASS: 2000x2000 MatMul executed in {dt:.2f} ms. Result sum = {c.sum().item():.2f}")
        return True
    except Exception as e:
        print(f"  FAIL: MatMul test failed: {e}")
        return False


def test_model_forward(device: torch.device, img_size: int = 1280) -> bool:
    print(f"\n[TEST 2] Baseline model forward pass on {device} (size={img_size}x{img_size})...")
    try:
        from research.models.baseline_fasterrcnn import build_baseline_model
        model = build_baseline_model(pretrained_backbone=False)
        model = model.to(device).eval()

        dummy_img = torch.rand(3, img_size, img_size, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = model([dummy_img])
            else:
                out = model([dummy_img])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) * 1000

        boxes_n = len(out[0]["boxes"])
        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0
        print(f"  PASS: Baseline Faster R-CNN forward pass OK ({dt:.1f} ms, {boxes_n} dummy predictions, peak VRAM {mem_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAIL: Model forward pass failed: {e}")
        return False


def test_batch_sizes(device: torch.device, img_size: int = 1280) -> None:
    if device.type != "cuda":
        print("\n[BATCH TEST] Skipped (CUDA not available)")
        return

    print(f"\n[BATCH TEST] Testing VRAM usage for img_size={img_size} at various batch sizes...")
    from research.models.baseline_fasterrcnn import build_baseline_model
    model = build_baseline_model(pretrained_backbone=False)
    model = model.to(device).eval()

    for batch_size in [1, 2, 4, 8, 16, 32]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            imgs = [torch.rand(3, img_size, img_size, device=device) for _ in range(batch_size)]
            t0 = time.perf_counter()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = model(imgs)
            torch.cuda.synchronize(device)
            dt = (time.perf_counter() - t0) * 1000
            mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"  Batch size {batch_size:2d}:  PASS  ({dt:.1f} ms, Peak VRAM: {mem_gb:.2f} GB)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch size {batch_size:2d}:  OOM   (Out of Memory)")
                torch.cuda.empty_cache()
                break
            else:
                print(f"  Batch size {batch_size:2d}:  FAIL  ({e})")
                break


def main():
    parser = argparse.ArgumentParser(description="HPC Hardware & Environment Validator")
    parser.add_argument("--batch_test", action="store_true", help="Run VRAM batch size stress test")
    parser.add_argument("--img_size", type=int, default=1280, help="Image size for forward pass test")
    args = parser.parse_args()

    cuda_ok = check_environment()
    dev = torch.device("cuda:0" if cuda_ok else "cpu")

    t_ok = test_tensor_ops(dev)
    m_ok = test_model_forward(dev, img_size=args.img_size)

    if args.batch_test:
        test_batch_sizes(dev, img_size=args.img_size)

    print("\n" + "=" * 60)
    if cuda_ok and t_ok and m_ok:
        print("RESULT: HPC ENVIRONMENT VALIDATION PASSED SUCCESSFULLY")
    else:
        print("RESULT: ENVIRONMENT VALIDATION COMPLETED WITH WARNINGS/ERRORS")
    print("=" * 60)


if __name__ == "__main__":
    main()
