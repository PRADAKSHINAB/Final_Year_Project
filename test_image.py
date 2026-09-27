from ultralytics import YOLO
from tkinter import Tk, filedialog
import os

# ============================================================
# 1. LOAD YOUR TRAINED MODEL
# ============================================================

MODEL_PATH = "runs/detect/runs/detect/baseline_yolo11s/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print("ERROR: Trained model not found!")
    print("Check this path:")
    print(MODEL_PATH)
    exit()

print("=" * 60)
print("Loading trained YOLO11s model...")
print("=" * 60)

model = YOLO(MODEL_PATH)

# ============================================================
# 2. SELECT AN IMAGE
# ============================================================

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an image to test",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    exit()

print("\nSelected image:")
print(image_path)

# ============================================================
# 3. RUN OBJECT DETECTION
# ============================================================

print("\nRunning YOLO detection...")

results = model.predict(
    source=image_path,
    imgsz=512,
    conf=0.25,
    device=0,
    save=True,
    show=True
)

# ============================================================
# 4. DISPLAY DETECTION RESULTS
# ============================================================

result = results[0]

print("\n" + "=" * 60)
print("DETECTION RESULTS")
print("=" * 60)

if len(result.boxes) == 0:
    print("No objects detected.")
else:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        print(
            f"{class_name:20s} "
            f"Confidence: {confidence:.2f}"
        )

print("\n" + "=" * 60)
print("Detection completed!")
print("=" * 60)

print("\nThe detected image has been saved inside:")
print("runs/detect/predict/")