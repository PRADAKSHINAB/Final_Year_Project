from ultralytics import YOLO

# Load the weights from your completed 1-epoch training
model = YOLO("runs/detect/train-2/weights/last.pt")

model.train(
    data="VisDrone.yaml",
    epochs=20,
    imgsz=512,
    batch=2,
    device=0,
    workers=0,
    amp=True,
    optimizer="auto",
    project="runs/detect",
    name="baseline_yolo11s",
    exist_ok=True
)