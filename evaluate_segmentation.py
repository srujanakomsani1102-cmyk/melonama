import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.segmentation.ultralight_vmunet.UltraLight_VM_UNet import UltraLight_VM_UNet

IMAGE_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1-2_Training_Input"
MASK_DIR = "/mnt/d/pytest_cache/datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth"
MODEL_PATH = "/mnt/d/pytest_cache/models/segmentation/UltraLight_VM_UNet.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = UltraLight_VM_UNet(
    num_classes=1,
    input_channels=3,
    c_list=[8, 16, 24, 32, 48, 64]
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

state_dict = {
    k.replace("module.", "", 1) if k.startswith("module.") else k: v
    for k, v in state_dict.items()
}

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

model = model.to(DEVICE)
model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def calculate_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

    return dice, iou, precision, recall, specificity, accuracy


image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
])

print("Images found:", len(image_files))

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []
specificity_scores = []
accuracy_scores = []

with torch.no_grad():

    for idx, filename in enumerate(image_files):

        image_id = os.path.splitext(filename)[0]
        mask_filename = image_id + "_segmentation.png"

        image_path = os.path.join(IMAGE_DIR, filename)
        mask_path = os.path.join(MASK_DIR, mask_filename)

        if not os.path.exists(mask_path):
            print("Missing mask:", filename)
            continue

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        mask = mask.resize((224, 224), Image.NEAREST)
        mask = np.array(mask)

        # Ground truth: non-zero pixels = lesion
        target = mask > 0

        output = model(image_tensor)

        # Model output is already probability-like
        prediction = output.squeeze().cpu().numpy()

        pred = prediction >= 0.5

        metrics = calculate_metrics(pred, target)

        dice, iou, precision, recall, specificity, accuracy = metrics

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        specificity_scores.append(specificity)
        accuracy_scores.append(accuracy)

        if (idx + 1) % 100 == 0:
            print(
                f"[{idx + 1}/{len(image_files)}] "
                f"Dice={dice:.4f} "
                f"IoU={iou:.4f}"
            )

print("\n" + "=" * 60)
print("ULTRALIGHT VM-UNET SEGMENTATION RESULTS")
print("=" * 60)

print(f"Images evaluated : {len(dice_scores)}")
print(f"Dice             : {np.mean(dice_scores):.4f} ({np.mean(dice_scores)*100:.2f}%)")
print(f"IoU              : {np.mean(iou_scores):.4f} ({np.mean(iou_scores)*100:.2f}%)")
print(f"Precision        : {np.mean(precision_scores):.4f} ({np.mean(precision_scores)*100:.2f}%)")
print(f"Recall           : {np.mean(recall_scores):.4f} ({np.mean(recall_scores)*100:.2f}%)")
print(f"Specificity      : {np.mean(specificity_scores):.4f} ({np.mean(specificity_scores)*100:.2f}%)")
print(f"Pixel Accuracy   : {np.mean(accuracy_scores):.4f} ({np.mean(accuracy_scores)*100:.2f}%)")

print("=" * 60)
