#!/bin/bash
#SBATCH --job-name=hismd_ablation_v2
#SBATCH --output=ablation_v2_slurm_%j.log
#SBATCH --error=ablation_v2_slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

# Activate Python Environment
if [ -f "/opt/venvs/cv/bin/python" ]; then
    PYTHON="/opt/venvs/cv/bin/python"
else
    PYTHON="python"
fi

cd "$(dirname "$0")" || exit 1

echo "=============================================================================="
echo " Starting SLURM Job: $SLURM_JOB_ID on node $(hostname)"
echo " CUDA Device: $CUDA_VISIBLE_DEVICES"
echo " Python: $PYTHON"
echo "=============================================================================="

# Run full 100-epoch ablation suite
$PYTHON train_ablation_v2.py --variant all --epochs 100
