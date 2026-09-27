import os
import cv2
from pathlib import Path

# -----------------------------
# Dataset Paths
# -----------------------------
BASE_DIR = Path("C:/visdrone2019")

TRAIN_IMG = BASE_DIR / "VisDrone2019-DET-train/VisDrone2019-DET-train/images"
TRAIN_ANN = BASE_DIR / "VisDrone2019-DET-train/VisDrone2019-DET-train/annotations"

VAL_IMG = BASE_DIR / "VisDrone2019-DET-val/VisDrone2019-DET-val/images"
VAL_ANN = BASE_DIR / "VisDrone2019-DET-val/VisDrone2019-DET-val/annotations"

OUTPUT_DIR = BASE_DIR / "datasets"

# VisDrone class mapping
CLASS_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9   # motor
}

def convert_dataset(image_dir, ann_dir, split):

    image_output = OUTPUT_DIR / split / "images"
    label_output = OUTPUT_DIR / split / "labels"

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.jpg"))

    print(f"\nConverting {split} dataset...")

    for image_path in images:

        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]

        annotation_file = ann_dir / (image_path.stem + ".txt")
        output_file = label_output / (image_path.stem + ".txt")

        with open(annotation_file, "r") as f:
            lines = f.readlines()

        yolo_labels = []

        for line in lines:

            data = line.strip().split(",")

            if len(data) < 8:
                continue

            left = float(data[0])
            top = float(data[1])
            width = float(data[2])
            height = float(data[3])

            score = int(data[4])
            category = int(data[5])

            # Ignore invalid objects
            if score == 0:
                continue

            if category not in CLASS_MAP:
                continue

            x_center = (left + width / 2) / w
            y_center = (top + height / 2) / h
            width /= w
            height /= h

            class_id = CLASS_MAP[category]

            yolo_labels.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        with open(output_file, "w") as f:
            f.write("\n".join(yolo_labels))

        destination = image_output / image_path.name

        if not destination.exists():
            os.link(image_path, destination)

    print(f"{split} conversion completed.")


convert_dataset(TRAIN_IMG, TRAIN_ANN, "train")
convert_dataset(VAL_IMG, VAL_ANN, "val")

print("\nFinished Successfully!")