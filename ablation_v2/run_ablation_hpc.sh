#!/bin/bash
# ==============================================================================
# HiSMD-Net Ablation Study V2 Runner Script (Linux / HPC)
# ==============================================================================

# Select python executable (prefer HPC CV environment if available)
if [ -f "/opt/venvs/cv/bin/python" ]; then
    PYTHON="/opt/venvs/cv/bin/python"
else
    PYTHON="python"
fi

echo "=============================================================================="
echo " Launching HiSMD-Net Ablation Study V2 (100 Epochs per Variant)"
echo " Using Python: $PYTHON"
echo "=============================================================================="

# 1. Run Pre-flight Validation
$PYTHON validate_setup.py
if [ $? -ne 0 ]; then
    echo "[!] Validation checks failed. Aborting full runs."
    exit 1
fi

# 2. Run All 7 Variants Sequentially for 100 Epochs
echo "[+] Starting 100-Epoch Ablation Training for Variants 1 to 7 ..."
$PYTHON train_ablation_v2.py --variant all --epochs 100

echo "=============================================================================="
echo " Ablation experiments complete. Check results in ablation_v2/:"
echo "   - ablation_v2/ablation_comparison.csv"
echo "   - ablation_v2/epoch_comparison.csv"
echo "   - ablation_v2/results_summary.txt"
echo "=============================================================================="
