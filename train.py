import os
import multiprocessing
from ultralytics import YOLO


def main():

    print("=" * 60)
    print("Loading YOLO11s model...")
    print("=" * 60)

    checkpoint = "runs/detect/baseline_yolo11s/weights/last.pt"

    if os.path.exists(checkpoint):
        print("Checkpoint found!")
        print("Resuming previous training...\n")

        model = YOLO(checkpoint)

        model.train(
            resume=True,
            workers=0
        )

    else:
        print("No checkpoint found.")
        print("Starting new baseline training...\n")

        model = YOLO("yolo11s.pt")

        model.train(

            # Dataset
            data="VisDrone.yaml",

            # Training
            epochs=10,
            imgsz=512,
            batch=2,

            # Hardware
            device=0,
            workers=0,
            amp=False,

            # Optimizer
            optimizer="auto",

            # Reproducibility
            seed=42,
            deterministic=True,

            # Output
            project="runs/detect",
            name="baseline_yolo11s",
            exist_ok=True,

            # Save
            save=True,
            plots=True,
            pretrained=True,
            verbose=True,

            # Early stopping
            patience=20
        )

    print("\n")
    print("=" * 60)
    print("Training Finished!")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
