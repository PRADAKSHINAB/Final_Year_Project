# ============================================================
# E1 BASELINE - SINGLE IMAGE PREDICTION
# ============================================================

import sys
import io
from pathlib import Path

# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
import torch
from PIL import Image, ImageDraw

import ipywidgets as widgets
from IPython.display import display, clear_output

from research.models.proposed_model import build_proposed_model
from research.models.baseline_fasterrcnn import load_checkpoint


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "research"
    / "results"
    / "E1_baseline"
    / "checkpoints"
    / "best.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.25


# ============================================================
# VISDRONE CLASS NAMES
# ============================================================

CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


# ============================================================
# LOAD E1 BASELINE MODEL
# ============================================================

print("=" * 70)
print("E1 BASELINE - SINGLE IMAGE PREDICTION")
print("=" * 70)

print("Project root :", PROJECT_ROOT)
print("Checkpoint   :", CHECKPOINT)
print("Device       :", DEVICE)

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        "\nE1 checkpoint was not found:\n"
        f"{CHECKPOINT}\n\n"
        "Please make sure E1 training has completed."
    )

print("\nCheckpoint found.")


print("\nLoading E1 Baseline Faster R-CNN...")


# E1 = baseline
# No Coordinate Attention
# No SACM

model, model_name = build_proposed_model(
    use_attention=False,
    use_sacm=False,
    attention_type="coordinate_attention",
    attention_location="fpn_lateral",
    attention_reduction=32,
    sacm_n_bins=4,
    sacm_context_channels=64,
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print("Loading trained checkpoint...")

load_checkpoint(
    model,
    str(CHECKPOINT),
    device=DEVICE
)

model = model.to(DEVICE)

model.eval()

print("\nModel loaded successfully!")
print("Model:", model_name)

print("=" * 70)


# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_image(image_bytes):

    # --------------------------------------------------------
    # Open uploaded image
    # --------------------------------------------------------

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    original_image = image.copy()


    # --------------------------------------------------------
    # Convert PIL image to PyTorch tensor
    # --------------------------------------------------------

    image_array = np.array(image)

    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    image_tensor = image_tensor.to(DEVICE)


    # --------------------------------------------------------
    # Run Faster R-CNN
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            [image_tensor]
        )[0]


    # --------------------------------------------------------
    # Get prediction results
    # --------------------------------------------------------

    boxes = prediction["boxes"].detach().cpu()

    labels = prediction["labels"].detach().cpu()

    scores = prediction["scores"].detach().cpu()


    # --------------------------------------------------------
    # Draw predictions
    # --------------------------------------------------------

    output_image = original_image.copy()

    draw = ImageDraw.Draw(output_image)

    detections = []


    # --------------------------------------------------------
    # Process every detection
    # --------------------------------------------------------

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        score = float(score)


        # Ignore low-confidence predictions

        if score < CONFIDENCE_THRESHOLD:
            continue


        # Bounding box coordinates

        x1, y1, x2, y2 = [
            int(v) for v in box
        ]


        # Class ID

        class_id = int(label)


        # Convert class ID to class name

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}"
        )


        # Save detection information

        detections.append(
            {
                "class_name": class_name,
                "class_id": class_id,
                "confidence": score,
                "box": (x1, y1, x2, y2)
            }
        )


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )


        # ----------------------------------------------------
        # Draw label
        # ----------------------------------------------------

        text = f"{class_name} {score:.2f}"


        try:

            text_box = draw.textbbox(
                (x1, y1),
                text
            )

            draw.rectangle(
                text_box,
                fill="red"
            )

        except Exception:

            pass


        draw.text(
            (x1, y1),
            text,
            fill="white"
        )


    return output_image, detections


# ============================================================
# CREATE UPLOAD BUTTON
# ============================================================

upload_button = widgets.FileUpload(
    accept="image/*",
    multiple=False,
    description="Upload Image"
)


# Output area

output_area = widgets.Output()


# ============================================================
# HANDLE UPLOADED IMAGE
# ============================================================

def handle_upload(change):

    with output_area:

        clear_output(wait=True)


        # ----------------------------------------------------
        # Check whether an image was uploaded
        # ----------------------------------------------------

        if not upload_button.value:

            return


        # ----------------------------------------------------
        # JupyterLab / ipywidgets compatibility
        #
        # Some versions return:
        #
        # tuple
        #
        # while older versions return:
        #
        # dictionary
        # ----------------------------------------------------

        value = upload_button.value


        if isinstance(value, tuple):

            uploaded_file = value[0]

        elif isinstance(value, dict):

            uploaded_file = list(
                value.values()
            )[0]

        else:

            print(
                "Unsupported upload format:",
                type(value)
            )

            return


        # ----------------------------------------------------
        # Get filename and bytes
        # ----------------------------------------------------

        filename = uploaded_file["name"]

        image_bytes = uploaded_file["content"]


        print("=" * 70)

        print("IMAGE UPLOADED")

        print("=" * 70)

        print("Filename:", filename)


        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        try:

            result_image, detections = predict_image(
                image_bytes
            )


            # ------------------------------------------------
            # Print number of detections
            # ------------------------------------------------

            print(
                "\nDetected objects:",
                len(detections)
            )

            print(
                "Confidence threshold:",
                CONFIDENCE_THRESHOLD
            )


            # ------------------------------------------------
            # Print detection details
            # ------------------------------------------------

            if len(detections) > 0:

                print("\nPredictions:")

                print("-" * 60)


                for i, detection in enumerate(
                    detections,
                    start=1
                ):

                    print(
                        f"{i:3d}. "
                        f"{detection['class_name']:<20} "
                        f"confidence = "
                        f"{detection['confidence']:.2f}"
                    )


            else:

                print(
                    "\nNo objects detected above "
                    f"confidence {CONFIDENCE_THRESHOLD}."
                )


            # ------------------------------------------------
            # Display result image
            # ------------------------------------------------

            print("\nPrediction image:")

            print("-" * 60)

            display(result_image)


        except Exception as e:

            print("\nERROR DURING PREDICTION")

            print("-" * 60)

            print("Error type:", type(e).__name__)

            print("Error:", str(e))


# ============================================================
# CONNECT BUTTON TO FUNCTION
# ============================================================

upload_button.observe(
    handle_upload,
    names="value"
)


# ============================================================
# DISPLAY INTERFACE
# ============================================================

print("\n")

print("E1 Baseline prediction is ready.")

print(
    "Click 'Upload Image' and select a drone image "
    "from your computer."
)

print("\n")


display(upload_button)

display(output_area)
