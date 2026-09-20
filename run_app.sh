#!/bin/bash
# ================================================================
# HiSMD-Net Web Application Launcher (Linux / HPC / Colab)
# AP50: 68.43%  mAP: 47.80%  APS: 30.23%
# ================================================================

echo "============================================================"
echo " HiSMD-Net Web Application Launcher"
echo " AP50: 68.43% | mAP: 47.80% | APS: 30.23%"
echo "============================================================"
echo ""

echo "[1/2] Checking dependencies..."
pip install -q fastapi uvicorn python-multipart torch torchvision opencv-python numpy pillow

echo "[2/2] Launching server on http://0.0.0.0:8000 ..."
python app.py
