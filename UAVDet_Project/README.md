# Small Object Detection in UAV Imagery Based on UAVDet


## 📌 Overview

Small object detection in UAV (Unmanned Aerial Vehicle) aerial images is a challenging computer vision task due to tiny object sizes, dense object distribution, varying object scales, and complex backgrounds. Conventional CNN-based detectors often lose fine-grained spatial information during feature extraction, while Transformer-based approaches usually require high computational resources.

This project focuses on reproducing the UAVDet framework for small object detection and investigating further improvements to enhance detection accuracy while maintaining real-time performance.

---

## 🎯 Objectives

- Improve small object detection in UAV aerial imagery.
- Preserve fine-grained feature information.
- Capture both local and global contextual information.
- Reduce false detections in complex environments.
- Maintain computational efficiency for real-time deployment.

---

## 📖 Base Paper

**UAVDet: A CNN–Mamba Hybrid Network for Efficient Small Object Detection in UAV Imagery**

Published in:

> Computer Vision and Image Understanding (Elsevier)

---

## 🚀 Current Progress

- ✅ Literature Survey Completed
- ✅ Base Paper Selected (UAVDet)
- ✅ VisDrone2019 Dataset Collected
- ✅ Development Environment Configured
- ✅ Baseline Implementation Using YOLOv11 Completed
- 🔄 Faster R-CNN Implementation In Progress
- 🔄 Performance Comparison Planned
- 🔄 Proposed Enhancement Under Investigation

---

## 🏗️ Current Architecture

```
Input UAV Image
        │
        ▼
Image Preprocessing
        │
        ▼
YOLOv11 Backbone
        │
        ▼
Feature Extraction
        │
        ▼
Neck (Feature Fusion)
        │
        ▼
Detection Head
        │
        ▼
Bounding Boxes + Class Predictions
```

---

## 💡 Planned Enhancement

The project aims to extend the baseline model after performance evaluation.

Potential enhancements include:

- Dynamic Attention Module
- Multi-Scale Feature Fusion
- Improved Tiny Object Feature Representation
- Faster R-CNN Performance Comparison
- Enhanced Localization Strategy
- Lightweight Feature Enhancement Module

> **Note:** These enhancements are currently under investigation and have not yet been integrated into the implementation.

---

## 📂 Dataset

### Current Dataset

- VisDrone2019-DET

### Future Evaluation (Planned)

- UAVDT
- DroneVehicle

---

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- YOLOv11
- Faster R-CNN (Ongoing)
- Computer Vision
- Deep Learning

---

## 📊 Evaluation Metrics

The model will be evaluated using:

- Precision
- Recall
- mAP
- AP50
- AP75
- APs (Small Object AP)
- FPS (Frames Per Second)

---

## 📈 Expected Outcomes

- Improved detection of tiny objects
- Better localization accuracy
- Reduced false detections
- Improved feature representation
- Efficient real-time inference

---

## 📌 Applications

The proposed framework can be applied to:

- Traffic Monitoring
- Smart City Surveillance
- Border Security
- Disaster Response
- Search and Rescue
- Wildlife Monitoring
- Infrastructure Inspection

---

## 🔬 Research Direction

This project reproduces the UAVDet framework and investigates additional techniques to further improve small object detection in UAV imagery. Future work will focus on integrating lightweight feature enhancement techniques and evaluating alternative detection architectures to improve both detection accuracy and computational efficiency.

---

## 👩‍💻 Project Team

**Department of Computer Science and Engineering**

Kongu Engineering College

---

## 📜 License

This project is developed for academic and research purposes only.
