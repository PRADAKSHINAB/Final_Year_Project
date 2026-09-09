# HPC Setup & Execution Guide — UAV Small-Object Detection

This guide outlines the exact, step-by-step procedure to deploy, verify, train, evaluate, and test the **SOA-FasterRCNN** UAV small-object detection system on an HPC Linux GPU cluster (e.g. NVIDIA H200 / A100 / V100).

---

## 1. Upload & Verify Archive

On your local machine, upload `uavdet_final_hpc_ready.zip` to your HPC directory.

On the HPC terminal:

```bash
# 1. Test ZIP file integrity
7z t uavdet_final_hpc_ready.zip
# or
unzip -t uavdet_final_hpc_ready.zip

# 2. Extract into project folder
unzip -q uavdet_final_hpc_ready.zip -d uavdet_system
cd uavdet_system
```

---

## 2. Environment & Dependencies

```bash
# 1. (Optional) Create or activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify all library imports
python research/check_dependencies.py

# 4. Verify GPU & VRAM capability
python research/check_hpc.py --batch_test
```

---

## 3. Dataset & Coordinate Integrity Audit

```bash
# Validate image files, COCO JSON schema, and label indexing
python research/validate_dataset.py

# Validate IoU math, augmentation pipeline, empty target handling, and coordinate rescaling
python research/test_coordinates_and_transforms.py
```

---

## 4. Quick Debug Test (~2 Minutes)

Before starting multi-epoch research training, run a lightweight end-to-end debug pass to verify CUDA forward/backward passes, loss convergence, validation, metric generation, and checkpoint saving:

```bash
python research/run_experiment.py --experiment baseline --mode debug --device cuda
```

---

## 5. Full Research Training Pipeline

Run the research experiments on your GPU:

### Option A: Run All Four Experiments Sequentially
Trains E1 $\rightarrow$ E2 $\rightarrow$ E3 $\rightarrow$ E4 with automatic cross-model comparison table and graphs:
```bash
python research/run_all_experiments.py --mode research --device cuda --resume
```

### Option B: Run Individual Experiments
```bash
# E1: Baseline Faster R-CNN (ResNet-50-FPN)
python research/run_experiment.py --experiment baseline --mode research --device cuda

# E2: Faster R-CNN + Coordinate Attention
python research/run_experiment.py --experiment attention --mode research --device cuda

# E3: Faster R-CNN + Scale-Aware Context Module (SACM)
python research/run_experiment.py --experiment second_module --mode research --device cuda

# E4: Full Proposed SOA-FasterRCNN (E2 + E3)
python research/run_experiment.py --experiment final_model --mode research --device cuda
```

---

## 6. Post-Training Evaluation & Summary Generation

Generate cross-experiment comparison charts and tables:

```bash
python research/evaluate_all.py
```

Outputs generated:
* `results/experiment_comparison.csv`
* `results/experiment_comparison.json`
* `results/graphs/map50_comparison.png`
* `results/graphs/map50_95_comparison.png`
* `results/graphs/ap_small_comparison.png`
* `results/graphs/f1_comparison.png`
* `results/graphs/parameter_comparison.png`
* `results/graphs/fps_comparison.png`

---

## 7. Interactive Testing on Your Own UAV / Drone Images

### Single-Image Testing
```bash
# Interactive GUI / Terminal prompt
python research/test_image.py --checkpoint results/E4_full/checkpoints/best.pth

# Direct image path with confidence threshold
python research/test_image.py \
    --checkpoint results/E4_full/checkpoints/best.pth \
    --image path/to/my_drone_image.jpg \
    --confidence 0.40 \
    --device cuda
```

### Batch Directory Testing
```bash
python research/test_images.py \
    --checkpoint results/E4_full/checkpoints/best.pth \
    --input_dir path/to/drone_photos/ \
    --confidence 0.40 \
    --device cuda
```

### Interactive Web UI (Streamlit)
```bash
streamlit run research/app.py
```
