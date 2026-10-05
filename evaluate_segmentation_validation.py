import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1-2_Training_Input"
MASK_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth"

MODEL_PATH = "/mnt/d/pytest_cache/models/segmentation/UltraLight_VM_UNet.pth"

SPLIT_PATH = "/mnt/d/pytest_cache/outputs/segmentation/segmentation_split.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD VALIDATION SPLIT
# ============================================================

with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)

validation_images = split_data["validation_images"]

print(
    "Validation images:",
    len(validation_images)
)


# ============================================================
# MODEL
# ============================================================

model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8, 16, 24, 32, 48, 64]
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove DataParallel prefix
state_dict = {
    k.replace("module.", "", 1)
    if k.startswith("module.")
    else k: v

    for k, v in state_dict.items()
}


missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))


model = model.to(DEVICE)
model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        pred,
        target
    ).sum()

    tn = np.logical_and(
        ~pred,
        ~target
    ).sum()

    fp = np.logical_and(
        pred,
        ~target
    ).sum()

    fn = np.logical_and(
        ~pred,
        target
    ).sum()


    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + 1e-7
    )


    iou = (
        tp
    ) / (
        tp + fp + fn + 1e-7
    )


    precision = (
        tp
    ) / (
        tp + fp + 1e-7
    )


    recall = (
        tp
    ) / (
        tp + fn + 1e-7
    )


    specificity = (
        tn
    ) / (
        tn + fp + 1e-7
    )


    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + 1e-7
    )


    return (
        dice,
        iou,
        precision,
        recall,
        specificity,
        accuracy
    )


# ============================================================
# THRESHOLDS
# ============================================================

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]


results = {}

for threshold in thresholds:

    results[threshold] = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "accuracy": []
    }


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for idx, filename in enumerate(
        validation_images
    ):

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )


        image_id = os.path.splitext(
            filename
        )[0]


        mask_path = os.path.join(
            MASK_DIR,
            image_id + "_segmentation.png"
        )


        if not os.path.exists(image_path):

            print(
                "Missing image:",
                filename
            )

            continue


        if not os.path.exists(mask_path):

            print(
                "Missing mask:",
                filename
            )

            continue


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        image_tensor = image_transform(
            image
        ).unsqueeze(0).to(DEVICE)


        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")


        mask = mask.resize(
            (224, 224),
            Image.NEAREST
        )


        mask = np.array(mask)

        target = mask > 0


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        output = model(
            image_tensor
        )


        # IMPORTANT:
        # This checkpoint already produces
        # probability-like output.

        prediction = (
            output
            .squeeze()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # TEST ALL THRESHOLDS
        # ----------------------------------------------------

        for threshold in thresholds:

            pred = (
                prediction >= threshold
            )


            metrics = calculate_metrics(
                pred,
                target
            )


            dice, iou, precision, recall, specificity, accuracy = metrics


            results[threshold]["dice"].append(
                dice
            )

            results[threshold]["iou"].append(
                iou
            )

            results[threshold]["precision"].append(
                precision
            )

            results[threshold]["recall"].append(
                recall
            )

            results[threshold]["specificity"].append(
                specificity
            )

            results[threshold]["accuracy"].append(
                accuracy
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (idx + 1) % 50 == 0:

            print(
                f"[{idx + 1}/{len(validation_images)}]"
            )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 80)
print("ISIC2018 VALIDATION RESULTS")
print("=" * 80)

print(
    "Validation images evaluated:",
    len(validation_images)
)

print()


# ============================================================
# PRINT RESULTS
# ============================================================

best_threshold = None
best_dice = -1


for threshold in thresholds:

    mean_dice = np.mean(
        results[threshold]["dice"]
    )

    mean_iou = np.mean(
        results[threshold]["iou"]
    )

    mean_precision = np.mean(
        results[threshold]["precision"]
    )

    mean_recall = np.mean(
        results[threshold]["recall"]
    )

    mean_specificity = np.mean(
        results[threshold]["specificity"]
    )

    mean_accuracy = np.mean(
        results[threshold]["accuracy"]
    )


    print(
        f"Threshold {threshold:.1f}"
    )

    print(
        f"  Dice        : {mean_dice:.4f}"
        f" ({mean_dice * 100:.2f}%)"
    )

    print(
        f"  IoU         : {mean_iou:.4f}"
        f" ({mean_iou * 100:.2f}%)"
    )

    print(
        f"  Precision   : {mean_precision:.4f}"
        f" ({mean_precision * 100:.2f}%)"
    )

    print(
        f"  Recall      : {mean_recall:.4f}"
        f" ({mean_recall * 100:.2f}%)"
    )

    print(
        f"  Specificity : {mean_specificity:.4f}"
        f" ({mean_specificity * 100:.2f}%)"
    )

    print(
        f"  Accuracy    : {mean_accuracy:.4f}"
        f" ({mean_accuracy * 100:.2f}%)"
    )

    print()


    if mean_dice > best_dice:

        best_dice = mean_dice
        best_threshold = threshold


# ============================================================
# BEST RESULT
# ============================================================

print("=" * 80)
print("BEST VALIDATION RESULT")
print("=" * 80)

print(
    f"Best threshold : {best_threshold:.1f}"
)

print(
    f"Best Dice      : {best_dice:.4f}"
    f" ({best_dice * 100:.2f}%)"
)

print("=" * 80)
