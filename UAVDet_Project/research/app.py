"""
research/app.py
===============
Interactive Streamlit Web Interface for UAV Small-Object Detection.

Features:
  1. Select Model Experiment Architecture (Baseline, Coordinate Attention, SACM, Full SOA-FasterRCNN).
  2. Upload any UAV/drone image (JPG, PNG, WEBP, BMP).
  3. Interactive Confidence Threshold slider.
  4. Real-time inference with detection visualization (bounding boxes, VisDrone class names, confidence).
  5. Detections data table and summary statistics (number of detections, inference time, device).

Usage:
  streamlit run research/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from research.models.proposed_model import build_proposed_model
from research.test_image import _cls_name, draw_detections, EXPERIMENT_MAP, VISDRONE_CLASSES

try:
    import streamlit as st
except ImportError:
    print("[ERROR] Streamlit is not installed. Run: pip install streamlit")
    sys.exit(1)


@st.cache_resource
def get_model(experiment_name: str, checkpoint_path: str, device_str: str):
    flags = EXPERIMENT_MAP.get(experiment_name, EXPERIMENT_MAP["final_model"])
    model, model_name = build_proposed_model(
        use_attention=flags["use_attention"],
        use_sacm=flags["use_sacm"],
        pretrained_backbone=False,
    )
    dev = torch.device(device_str)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    state = ckpt.get("model", ckpt)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    model = model.to(dev).eval()
    return model, dev


def main():
    st.set_page_config(
        page_title="UAV Small-Object Detection",
        page_icon="🚁",
        layout="wide",
    )

    st.title("🚁 UAV Small-Object Detection System")
    st.markdown(
        "**Research Project:** Small-Object Detection in UAV Imagery using VisDrone-DET & SOA-FasterRCNN."
    )

    # Sidebar settings
    st.sidebar.header("Configuration")
    exp_choice = st.sidebar.selectbox(
        "Select Experiment Model",
        options=["final_model", "attention", "second_module", "baseline"],
        format_func=lambda x: {
            "baseline": "E1: Baseline Faster R-CNN",
            "attention": "E2: Faster R-CNN + Coordinate Attention",
            "second_module": "E3: Faster R-CNN + SACM",
            "final_model": "E4: Full Proposed SOA-FasterRCNN (E2 + E3)",
        }.get(x, x),
    )

    default_ckpt = str(PROJECT_ROOT / f"results/{exp_choice}/checkpoints/best.pth")
    ckpt_path = st.sidebar.text_input("Checkpoint Path (.pth)", value=default_ckpt)

    conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.50, step=0.05)
    img_size = st.sidebar.selectbox("Input Resolution", options=[640, 1280], index=1)
    use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=torch.cuda.is_available())
    device_str = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Hardware Device:** `{device_str.upper()}`")

    # Main Area
    uploaded_file = st.file_uploader("Upload UAV / Drone Image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        if img_bgr is None:
            st.error("Failed to decode the uploaded image.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("Detect Objects 🚀", type="primary"):
            if not Path(ckpt_path).exists():
                st.error(f"Checkpoint file not found: `{ckpt_path}`. Please train the model or specify a valid checkpoint.")
                return

            with st.spinner(f"Running inference on {device_str.upper()}..."):
                model, dev = get_model(exp_choice, ckpt_path, device_str)

                orig_h, orig_w = img_bgr.shape[:2]
                img_resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(dev)

                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model([tensor])[0]
                dt_ms = (time.perf_counter() - t0) * 1000

                keep = out["scores"].cpu() >= conf_thresh
                boxes = out["boxes"].cpu()[keep].numpy().tolist()
                labels = out["labels"].cpu()[keep].numpy().tolist()
                scores = out["scores"].cpu()[keep].numpy().tolist()

                scale_x = orig_w / img_size
                scale_y = orig_h / img_size
                boxes_orig = [
                    [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                    for b in boxes
                ]

                annotated = draw_detections(img_bgr, boxes_orig, labels, scores)

            with col2:
                st.subheader("Detections")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Metrics row
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects Detected", len(boxes_orig))
            m2.metric("Inference Time", f"{dt_ms:.1f} ms")
            m3.metric("Hardware Device", device_str.upper())

            # Detections table
            if boxes_orig:
                table_data = []
                for box, lbl, score in sorted(zip(boxes_orig, labels, scores), key=lambda x: -x[2]):
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    table_data.append({
                        "Class": _cls_name(lbl),
                        "Confidence": f"{score:.4f}",
                        "BBox (x1, y1, x2, y2)": f"[{x1}, {y1}, {x2}, {y2}]",
                        "Width": x2 - x1,
                        "Height": y2 - y1,
                    })
                st.subheader("Detection Details")
                st.table(table_data)
            else:
                st.info("No objects detected above the confidence threshold.")


if __name__ == "__main__":
    main()
