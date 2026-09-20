# HiSMD-Net: HPC Training & Resume Guide

## 1. Directory Structure on HPC
Place your files on the HPC server like this:
```
~/
+-- Dataset/
¦   +-- VisDrone2019-DET-train/
¦   ¦   +-- images/
¦   ¦   +-- annotations/
¦   +-- VisDrone2019-DET-val/
¦       +-- images/
¦       +-- annotations/
+-- train.py
+-- run_hpc.sh
+-- requirements_hpc.txt
```
*(Note: If VisDrone2019-DET-train is nested as `VisDrone2019-DET-train/VisDrone2019-DET-train/images`, the script automatically detects it!)*

## 2. How to Run Training

### Option A: Direct Run (Foreground)
```bash
python3 train.py
```

### Option B: Background Run with nohup (Recommended - runs even if you disconnect)
```bash
nohup python3 train.py > train.log 2>&1 &
```
To monitor training live:
```bash
tail -f train.log
```

## 3. How Auto-Resume Works
If training gets interrupted at any epoch (e.g. Epoch 25 or 90):
* The script saves `hismdet_output/last_hismdnet.pt` **every single epoch**.
* Simply run `python3 train.py` again!
* It will automatically detect `last_hismdnet.pt`, restore all model weights, optimizer state, and learning rate, and continue training from the next epoch seamlessly.

## 4. Where All Outputs are Saved
All outputs are organized in the `./hismdet_output/` folder:
* `best_hismdnet.pt`: Best model checkpoint based on AP50 / mAP
* `last_hismdnet.pt`: Most recent checkpoint for instant resumption
* `training_log.csv`: Loss and mAP progression for all epochs
* `vis_epoch_XXX.png`: Qualitative bounding box detection visual comparisons
* `predictions_txt/`: VisDrone format detection txt files
