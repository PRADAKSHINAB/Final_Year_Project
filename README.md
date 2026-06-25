Enhanced UAVDet: Dynamic Attention-Assisted Multi-Scale Mamba Fusion for Real-Time Small Object Detection in UAV Imagery

# Enhanced UAVDet: Dynamic Attention-Assisted Multi-Scale Mamba Fusion for Real-Time Small Object Detection in UAV Imagery

## 📌 Overview

Small object detection in UAV aerial images is a challenging computer vision task due to tiny object sizes, dense object distribution, and complex backgrounds. Traditional CNN-based detectors often lose fine-grained information, while Transformer-based methods require high computational resources.

This project proposes an enhanced version of **UAVDet**, a CNN-Mamba hybrid object detection framework, by integrating **Dynamic Attention** and **Multi-Scale Mamba Feature Fusion** to improve detection accuracy while maintaining real-time performance.

---

## 🎯 Objectives

- Improve small object detection accuracy in UAV imagery.
- Preserve fine-grained features during feature extraction.
- Capture both local and global contextual information.
- Reduce false detections in complex backgrounds.
- Maintain lightweight architecture for real-time applications.

---

## 🚀 Proposed Method

The proposed framework enhances the original UAVDet architecture by introducing:

- Dynamic Attention Module
- Multi-Scale Mamba Feature Fusion
- Improved Tiny Object Feature Representation
- Efficient Feature Pyramid Network
- Optimized Loss Function for better localization

---

## 🏗️ Architecture

Input Image
↓
CNN Backbone
↓
CSPMB (Cross Stage Partial Mamba Block)
↓
Dynamic Attention Module
↓
Multi-Scale Mamba Fusion
↓
Tiny-Focused Feature Pyramid Network (TFPN)
↓
Detection Head
↓
Bounding Box & Class Prediction

---

## ✨ Key Features

- CNN + Mamba Hybrid Architecture
- Dynamic Attention Mechanism
- Multi-Scale Feature Fusion
- Tiny Object Detection Head
- Lightweight Model
- Real-Time Detection
- Reduced False Positives
- Improved Detection Accuracy

---

## 📂 Dataset

The model is trained and evaluated using:

- VisDrone2019
- UAVDT
- DroneVehicle

---

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- CUDA
- Mamba
- Deep Learning
- Computer Vision

---

## 📊 Expected Outcomes

- Improved Average Precision (AP)
- Better Small Object AP (APs)
- Reduced False Detections
- Faster Inference Speed
- Real-Time UAV Deployment

---

## 📈 Evaluation Metrics

- Precision
- Recall
- mAP
- AP50
- AP75
- APs (Small Object AP)
- FPS (Frames Per Second)

---

## 📌 Applications

- Traffic Monitoring
- Smart City Surveillance
- Border Security
- Disaster Response
- Search and Rescue
- Wildlife Monitoring
- Infrastructure Inspection

---

## 🔬 Research Contribution

This work extends the UAVDet framework by integrating Dynamic Attention with Multi-Scale Mamba Fusion to improve feature representation for tiny objects while preserving real-time inference capability.

---

## 👩‍💻 Project Team

Final Year B.E. Computer Science and Engineering

Kongu Engineering College

---

## 📄 Base Paper

**UAVDet: A CNN–Mamba Hybrid Network for Efficient Small Object Detection in UAV Imagery**

Published in:
Computer Vision and Image Understanding (Elsevier), 2026.

---

## 📜 License

This project is developed for academic and research purposes.
I also recommend slightly refining the title for publication quality:

Enhanced UAVDet: Dynamic Attention-Assisted Multi-Scale Mamba Fusion for Efficient Small Object Detection in UAV Imagery
