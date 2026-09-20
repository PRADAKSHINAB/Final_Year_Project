# HiSMD-Net: Hierarchical Super-resolved Mamba Detector for UAV Small Object Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![VisDrone](https://img.shields.io/badge/Dataset-VisDrone2019--DET-blue?logo=drone)](https://github.com/VisDrone/VisDrone-Dataset)
[![AP50](https://img.shields.io/badge/AP50-68.43%25-emerald?style=flat-square)](./hismdet_output/training_log.csv)
[![mAP](https://img.shields.io/badge/mAP--50:95-47.80%25-60a5fa?style=flat-square)](./hismdet_output/training_log.csv)
[![APS](https://img.shields.io/badge/APS--Small-30.23%25-a78bfa?style=flat-square)](./hismdet_output/training_log.csv)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Final Year Engineering Research Project**  
> **Department of Computer Science and Engineering, Kongu Engineering College**  
> **Authors:** Prahadheesh S (`prahadheeshs.23cse`) & Team  
> **Trained on:** NVIDIA H200 NVL (HPC Cluster) · 100 Epochs

---

## 📌 Abstract

Unmanned Aerial Vehicle (UAV) object detection faces severe challenges due to extreme object scale variations, dense spatial clustering, and severe feature degradation of micro-objects ($<16\text{px}$). **HiSMD-Net** (*Hierarchical Super-resolved Mamba Detector*) introduces a novel multi-scale deep learning framework engineered specifically for high-altitude drone surveillance. By integrating sub-pixel spatial reconstruction, non-local Mamba state-space spatial attention, and a high-resolution Stride-2 feature pyramid, HiSMD-Net achieves state-of-the-art detection accuracy on the **VisDrone2019-DET** benchmark while maintaining real-time execution speeds ($16.49\text{M}$ parameters).

---

## ✨ Key Architectural Innovations

```
Input (640×640×3)
  ├── 1. ASRFR Sub-pixel Reconstruction Module (PixelUnshuffle → SR-Net → PixelShuffle)
  ├── 2. ResNet-50 Feature Backbone (Pretrained Stage 1–3)
  ├── 3. SpatialSSM (Mamba-style 7×7 Depthwise + Dual-Gated Pointwise Attention)
  ├── 4. 4-Scale Stride-2 Sub-Pixel FPN Pyramid (320×320, 160×160, 80×80, 40×40)
  └── 5. Decoupled Cls + Softplus-Bounded LTRB Offset Regression (136,000 Anchors)
```

1. **ASRFR (Adaptive Sub-pixel Resolution Feature Reconstructor)**: Recovers sub-pixel spatial details before backbone encoding using `PixelUnshuffle(2) → 3×Conv → PixelShuffle(2)`, restoring edges of micro-pedestrians and vehicles.
2. **SpatialSSM Block (Mamba-Style State-Space Attention)**: Employs a numerically stable $7\times 7$ depthwise convolution coupled with dual-gated GELU pointwise linear projections to capture global spatial dependencies across dense drone clusters without quadratic transformer complexity.
3. **Stride-2 Sub-Pixel Pyramid Head**: Reconstructs a ultra-dense $320\times 320$ feature grid (Stride 2) generating $102,400$ dedicated anchor points for tiny objects ($<16\text{px}$).
4. **Softplus Bounded LTRB Regression**: Uses Softplus-activated relative distance offset decoding clamped to $[0, 8\times\text{stride}]$, guaranteeing non-negative coordinates and numerical stability.

---

## 📊 Experimental Benchmark Results

Evaluated on the official **VisDrone2019-DET Validation Set** ($500$ aerial images, $10$ object categories) under standard COCO evaluation protocols ($\text{IoU}=0.50:0.95$):

### SOTA & Base Paper Comparison Table

| Model Architecture | Backbone / Neck | AP50 (%) | mAP (0.50:0.95) (%) | APS (Small $<32\text{px}$) (%) | APM (Medium) (%) | Parameters | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **HiSMD-Net (Ours)** | **ASRFR + ResNet-50 + SpatialSSM + Stride-2 FPN** | **68.43%** | **47.80%** | **30.23%** | **69.72%** | **16.49M** | **Proposed SOTA** |
| 📄 **UAVDet (Base Paper)** | **ResNet-50 + CSPMB Mamba + TFPN** | **64.20%** | **43.10%** | **24.50%** | **63.80%** | **19.80M** | **Base Paper Target** |
| YOLOv9-c | GELAN + P3-P5 Neck | 62.40% | 41.60% | 18.10% | 58.40% | 25.30M | Baseline |
| YOLOv8-m | CSPDarknet + PANet | 60.80% | 39.20% | 17.40% | 56.10% | 25.90M | Baseline |
| RT-DETR-R50 | ResNet-50 + Hybrid Encoder | 59.10% | 38.90% | 14.20% | 52.80% | 42.00M | Transformer |
| Faster R-CNN | ResNet-50 + FPN | 47.20% | 28.70% | 9.10% | 41.20% | 41.53M | Two-Stage |

> 📈 **Key Takeaway:** HiSMD-Net outperforms the original **UAVDet Base Paper** across all evaluation criteria:
> - **+4.23% higher AP50** ($68.43\%$ vs $64.20\%$)
> - **+4.70% higher mAP** ($47.80\%$ vs $43.10\%$)
> - **+5.73% higher Small Object Accuracy (APS)** ($30.23\%$ vs $24.50\%$)
> - **16.7% fewer parameters** ($16.49\text{M}$ vs $19.80\text{M}$) due to ASRFR sub-pixel feature efficiency.

---

## 🔬 Empirically Optimal Hyperparameters

* **Confidence Threshold ($\text{Conf}^* = 0.25$)**: Mathematically derived from the Precision-Recall curve inflection point on the VisDrone validation set. At $\tau = 0.25$, Precision ($72.4\%$) and Recall ($65.1\%$) achieve the global maximum Harmonic Mean **$F_1$-Score = 0.685**.
* **NMS IoU Threshold ($\text{IoU}^* = 0.50$)**: Aligned with official MS-COCO / VisDrone evaluation criteria to prevent duplicate anchor bounding boxes while preserving dense adjacent vehicle and pedestrian clusters.

---

## 🛸 Interactive Web Application Suite

The repository includes a full-stack, glassmorphic **Web Application Suite** for live model demonstration, image inference, and presentation:

```
web_application/
├── app.py                            # FastAPI Backend Engine
├── index.html                        # Pure-CSS Interactive Dashboard UI
├── run_app.bat                       # 1-Click Windows Launcher
├── run_app.sh                        # 1-Click Linux / HPC Launcher
└── sample_images/                    # Pre-loaded VisDrone Aerial Scene Presets
```

### 🌟 Web App Features:
- **Before / After Split Slider**: Interactively drag a split line to compare raw drone footage vs annotated bounding boxes.
- **Deep Zoom & Pan Canvas**: Smooth zoom up to $15\times$ to inspect micro-pedestrians ($<16\text{px}$).
- **One-Click Validation Presets**: Instant testing on pre-loaded VisDrone drone scenes.
- **Analytics & Export**: Per-class distribution charts, coordinate tables, CSV report export, and high-res PNG download.
- **JupyterHub & Firewall Proof**: Pure standalone CSS with zero external CDN dependencies and automatic XSRF token handling.

### How to Run the Web Application:

#### Windows PC:
Double-click **`web_application/run_app.bat`** or run:
```cmd
python app.py
```
Open **`http://localhost:8000`** in your browser.

#### HPC Cluster / Linux:
```bash
export PYTHONPATH=~/my_python_packages:$PYTHONPATH
/opt/venvs/cv/bin/python app.py
```
Access via JupyterHub Proxy: `https://hpc.kongu.edu/user/prahadheeshs.23cse/proxy/8000/`

---

## 📂 Repository Layout

```text
Final_Year_Project/
│
├── 🛸 web_application/                   # Full Interactive Web App Suite
│   ├── app.py                            # FastAPI Backend Server
│   ├── index.html                        # Glassmorphic Web Dashboard UI
│   ├── run_app.bat                       # Windows Launcher
│   ├── run_app.sh                        # Linux / HPC Launcher
│   └── sample_images/                    # VisDrone Validation Scene Presets
│
├── 🏆 hismdet_output/                    # Model Benchmark & Evaluation Logs
│   ├── training_log.csv                  # Full 100-Epoch Metrics (AP50: 68.43%)
│   ├── vis_epoch_005.png → 100.png       # Epoch Visualizations (Every 5 Epochs)
│   └── HiSMD_Net_Paper_Draft.docx        # Generated Research Paper Draft
│
├── 🧠 Training & Baseline Code
│   ├── train.py                          # 100-Epoch HiSMD-Net PyTorch Training Code
│   ├── project.py                        # Baseline Research Code & Warmup Scheduler
│   ├── draft.py                          # IEEE Paper Generator Script
│   ├── run_hpc.sh                        # HPC SLURM Script
│   └── HiSMD_Net_HPC_Code.zip            # Portable HPC Deployment Zip Package
│
└── 📚 Project Documentation & Presentations
    ├── Base Paper_compressed.pdf         # Base Reference Paper
    ├── PROPOSED METHODOLOGY              # Methodology Specification
    ├── RESEARCH GAP ANALYSIS             # Gap Analysis Specification
    ├── Research_Paper_Abstract_Introduction.docx
    ├── Review_1.pptx                     # Zeroth / Review 1 PPT
    ├── finalproject.pptx                 # Final Viva Presentation Slides
    └── literature survey                 # Literature Survey Specification
```

---

## 🛠️ Model Weights & Reproducibility

### Trained Weights Download:
Due to GitHub's $100\text{MB}$ single file limit, the $187\text{MB}$ trained PyTorch checkpoint is hosted on Google Drive:
- 💾 **`best_hismdnet.pt` (187 MB, Epoch 100)**: [Download Weights from Google Drive](https://drive.google.com/drive/u/0/my-drive)

### Training Reproduction:
```bash
# Clone Repository
git clone https://github.com/PRADAKSHINAB/Final_Year_Project.git
cd Final_Year_Project

# Run 100-Epoch HiSMD-Net Training
python train.py
```

---

## 📜 Citation & License

This project is licensed under the **MIT License**.

If you use **HiSMD-Net** in your research or project, please cite:

```bibtex
@article{prahadheesh2026hismdnet,
  title={HiSMD-Net: Hierarchical Super-resolved Mamba Detector for UAV Small Object Detection},
  author={Prahadheesh, S. and Team},
  journal={Department of Computer Science and Engineering, Kongu Engineering College},
  year={2026}
}
```
