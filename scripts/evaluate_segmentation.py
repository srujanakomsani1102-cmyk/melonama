"""
Evaluate the trained UNet on the SAME held-out validation split
created by train_segmentation.py.

Metrics:
    Dice
    IoU
    Precision
    Recall / Sensitivity
    Specificity
    Pixel Accuracy

Also saves:
    JSON summary
    CSV per-image metrics
    visual prediction examples
"""

import sys
import json
import csv
import argparse
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent
    )
)

import cv2
import numpy as np
import torch

from app.ml.segmentation.unet import UNet


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "ISIC2018"
)

IMAGE_DIR = (
    DATASET_DIR
    / "ISIC2018_Task1-2_Training_Input"
)

MASK_DIR = (
    DATASET_DIR
    / "ISIC2018_Task1_Training_GroundTruth"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "segmentation"
    / "unet_isic2018.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "segmentation"
)

SPLIT_PATH = (
    OUTPUT_DIR
    / "segmentation_split.json"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    ).astype(np.uint8)

    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    # True Positive

    tp = np.logical_and(
        prediction == 1,
        ground_truth == 1
    ).sum()

    # False Positive

    fp = np.logical_and(
        prediction == 1,
        ground_truth == 0
    ).sum()

    # False Negative

    fn = np.logical_and(
        prediction == 0,
        ground_truth == 1
    ).sum()

    # True Negative

    tn = np.logical_and(
        prediction == 0,
        ground_truth == 0
    ).sum()

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = (
        2 * tp
    ) / max(
        1,
        2 * tp + fp + fn
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (
        tp
    ) / max(
        1,
        tp + fp + fn
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp
    ) / max(
        1,
        tp + fp
    )

    # --------------------------------------------------------
    # Recall / Sensitivity
    # --------------------------------------------------------

    recall = (
        tp
    ) / max(
        1,
        tp + fn
    )

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity = (
        tn
    ) / max(
        1,
        tn + fp
    )

    # --------------------------------------------------------
    # Pixel Accuracy
    # --------------------------------------------------------

    accuracy = (
        tp + tn
    ) / max(
        1,
        tp + tn + fp + fn
    )

    return {

        "dice":
            float(dice),

        "iou":
            float(iou),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "sensitivity":
            float(recall),

        "specificity":
            float(specificity),

        "pixel_accuracy":
            float(accuracy),

        "tp":
            int(tp),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tn":
            int(tn)

    }


# ============================================================
# LOAD VALIDATION SPLIT
# ============================================================

def load_validation_pairs():

    if not SPLIT_PATH.exists():

        raise FileNotFoundError(

            "\nValidation split file not found:\n"

            f"{SPLIT_PATH}\n\n"

            "You must run train_segmentation.py first."
        )

    with open(
        SPLIT_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        split_data = json.load(
            file
        )

    validation_images = (
        split_data[
            "validation_images"
        ]
    )

    pairs = []

    missing = []

    # --------------------------------------------------------
    # Build mask lookup
    # --------------------------------------------------------

    masks = list(
        MASK_DIR.glob("*")
    )

    mask_lookup = {}

    for mask in masks:

        if not mask.is_file():
            continue

        stem = mask.stem

        if stem.endswith(
            "_segmentation"
        ):

            stem = stem[
                :-13
            ]

        mask_lookup[
            stem
        ] = mask

    # --------------------------------------------------------
    # Match validation images
    # --------------------------------------------------------

    for image_name in validation_images:

        image_path = (
            IMAGE_DIR
            / image_name
        )

        image_stem = (
            Path(image_name).stem
        )

        mask_path = (
            mask_lookup
            .get(image_stem)
        )

        if (
            image_path.exists()
            and mask_path is not None
        ):

            pairs.append(
                (
                    image_path,
                    mask_path
                )
            )

        else:

            missing.append(
                image_name
            )

    print(
        f"\nValidation images in split: "
        f"{len(validation_images)}"
    )

    print(
        f"Valid image-mask pairs: "
        f"{len(pairs)}"
    )

    if missing:

        print(
            f"Missing pairs: "
            f"{len(missing)}"
        )

        for name in missing[:10]:

            print(
                f"  {name}"
            )

    if not pairs:

        raise RuntimeError(
            "No validation image-mask pairs found."
        )

    return pairs


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    samples=0,
    threshold=0.5
):

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )

    print(
        "\n=========================================="
    )

    print(
        "UNET SEGMENTATION EVALUATION"
    )

    print(
        "=========================================="
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load validation images
    # --------------------------------------------------------

    pairs = load_validation_pairs()

    if samples > 0:

        pairs = pairs[
            :samples
        ]

    print(
        f"\nImages to evaluate: "
        f"{len(pairs)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = UNet(

        in_channels=3,

        out_channels=1,

        base_features=16

    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "\nModel loaded successfully."
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    visual_count = 0

    for index, (
        image_path,
        mask_path
    ) in enumerate(
        pairs,
        start=1
    ):

        print(
            f"Evaluating "
            f"[{index}/{len(pairs)}] "
            f"{image_path.name}"
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            print(
                "  Skipped: image unreadable"
            )

            continue

        # ----------------------------------------------------
        # Read ground truth
        # ----------------------------------------------------

        ground_truth = cv2.imread(

            str(mask_path),

            cv2.IMREAD_GRAYSCALE
        )

        if ground_truth is None:

            print(
                "  Skipped: mask unreadable"
            )

            continue

        original_height, original_width = (
            original.shape[:2]
        )

        ground_truth = (
            ground_truth > 127
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Resize image for UNet
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        resized = cv2.resize(

            rgb,

            (256, 256),

            interpolation=cv2.INTER_AREA
        )

        tensor = torch.from_numpy(

            resized.transpose(
                2,
                0,
                1
            )

        ).float().unsqueeze(0) / 255.0

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                tensor.to(device)
            )

            probability = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        predicted_256 = (

            probability >= threshold

        ).astype(np.uint8)

        # ----------------------------------------------------
        # Resize prediction to original size
        # ----------------------------------------------------

        prediction = cv2.resize(

            predicted_256,

            (
                original_width,
                original_height
            ),

            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(

            prediction,

            ground_truth

        )

        metrics[
            "image"
        ] = image_path.name

        results.append(
            metrics
        )

        print(
            f"  Dice: "
            f"{metrics['dice']:.4f} | "

            f"IoU: "
            f"{metrics['iou']:.4f} | "

            f"Recall: "
            f"{metrics['recall']:.4f}"
        )

        # ----------------------------------------------------
        # Save first 5 visualizations
        # ----------------------------------------------------

        if visual_count < 5:

            # Ground truth mask

            gt_visual = (
                ground_truth
                * 255
            ).astype(
                np.uint8
            )

            # Prediction mask

            pred_visual = (
                prediction
                * 255
            ).astype(
                np.uint8
            )

            # Convert to BGR

            gt_visual = cv2.cvtColor(

                gt_visual,

                cv2.COLOR_GRAY2BGR

            )

            pred_visual = cv2.cvtColor(

                pred_visual,

                cv2.COLOR_GRAY2BGR

            )

            # Overlay

            overlay = (
                original.copy()
            )

            gt_contours, _ = (
                cv2.findContours(

                    ground_truth,

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_SIMPLE
                )
            )

            pred_contours, _ = (
                cv2.findContours(

                    prediction,

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_SIMPLE
                )
            )

            # Ground truth = GREEN

            cv2.drawContours(

                overlay,

                gt_contours,

                -1,

                (0, 255, 0),

                2
            )

            # Prediction = RED

            cv2.drawContours(

                overlay,

                pred_contours,

                -1,

                (0, 0, 255),

                2
            )

            # Make same height

            original_small = cv2.resize(
                original,
                (256, 256)
            )

            gt_small = cv2.resize(
                gt_visual,
                (256, 256)
            )

            pred_small = cv2.resize(
                pred_visual,
                (256, 256)
            )

            overlay_small = cv2.resize(
                overlay,
                (256, 256)
            )

            panel = np.hstack(

                [
                    original_small,
                    gt_small,
                    pred_small,
                    overlay_small
                ]

            )

            output_image = (
                OUTPUT_DIR
                / f"segmentation_result_{visual_count + 1}.jpg"
            )

            cv2.imwrite(
                str(output_image),
                panel
            )

            visual_count += 1

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No images were successfully evaluated."
        )

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    metric_names = [

        "dice",

        "iou",

        "precision",

        "recall",

        "specificity",

        "pixel_accuracy"

    ]

    averages = {}

    for metric in metric_names:

        values = [
            result[metric]
            for result in results
        ]

        averages[metric] = float(
            np.mean(values)
        )

    # --------------------------------------------------------
    # Print final metrics
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "FINAL SEGMENTATION METRICS"
    )

    print(
        "=========================================="
    )

    print(
        f"Images evaluated : "
        f"{len(results)}"
    )

    print(
        f"Dice             : "
        f"{averages['dice']:.4f}"
    )

    print(
        f"IoU              : "
        f"{averages['iou']:.4f}"
    )

    print(
        f"Precision        : "
        f"{averages['precision']:.4f}"
    )

    print(
        f"Recall           : "
        f"{averages['recall']:.4f}"
    )

    print(
        f"Sensitivity      : "
        f"{averages['recall']:.4f}"
    )

    print(
        f"Specificity      : "
        f"{averages['specificity']:.4f}"
    )

    print(
        f"Pixel Accuracy   : "
        f"{averages['pixel_accuracy']:.4f}"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    summary = {

        "dataset":
            "ISIC2018",

        "evaluation":
            "held_out_validation",

        "number_of_images":
            len(results),

        "dice":
            averages["dice"],

        "iou":
            averages["iou"],

        "precision":
            averages["precision"],

        "recall":
            averages["recall"],

        "sensitivity":
            averages["recall"],

        "specificity":
            averages["specificity"],

        "pixel_accuracy":
            averages["pixel_accuracy"],

        "threshold":
            threshold,

        "model":
            str(MODEL_PATH)

    }

    json_path = (
        OUTPUT_DIR
        / "segmentation_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "segmentation_per_image_metrics.csv"
    )

    fields = [

        "image",

        "dice",

        "iou",

        "precision",

        "recall",

        "sensitivity",

        "specificity",

        "pixel_accuracy",

        "tp",

        "fp",

        "fn",

        "tn"

    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        "\nResults saved:"
    )

    print(
        json_path
    )

    print(
        csv_path
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help=(
            "Number of validation images. "
            "0 = all validation images."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    evaluate(

        samples=args.samples,

        threshold=args.threshold

    )