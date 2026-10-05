"""
Lesion segmentation service for CEFM (Rule 11).
Integrates deep UNet segmentation model trained on ISIC2018
with morphology-based classical fallback.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SEGMENTATION_WEIGHTS = BASE_DIR / "models" / "segmentation" / "unet_isic2018.pt"

_unet_model = None


def _get_unet_model():
    global _unet_model
    if _unet_model is not None:
        return _unet_model

    if SEGMENTATION_WEIGHTS.exists():
        try:
            from app.ml.segmentation.unet import UNet
            model = UNet(in_channels=3, out_channels=1, base_features=16)
            model.load_state_dict(torch.load(SEGMENTATION_WEIGHTS, map_location="cpu"))
            model.eval()
            _unet_model = model
            print(f"[segmentation] Loaded UNet weights from {SEGMENTATION_WEIGHTS}")
            return _unet_model
        except Exception as exc:
            print(f"[segmentation] Could not load UNet weights: {exc}")
            return None
    return None


def remove_hair(image_bgr: np.ndarray) -> np.ndarray:
    """
    DullRazor-style hair removal:
    blackhat morphology -> hair mask -> inpaint.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=2)

    if cv2.countNonZero(hair_mask) == 0:
        return image_bgr

    return cv2.inpaint(image_bgr, hair_mask, 5, cv2.INPAINT_TELEA)


def segment_lesion_classical(image: np.ndarray, exclude_polygon=None) -> np.ndarray:
    """Morphology + Otsu fallback segmentation."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    blurred = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if exclude_polygon is not None:
        poly = np.asarray(exclude_polygon, dtype=np.int32).reshape(-1, 2)
        if len(poly) >= 3:
            cv2.fillPoly(thresh, [poly], 0)

    if np.count_nonzero(thresh) > 0.6 * thresh.size:
        thresh = cv2.bitwise_not(thresh)

    if exclude_polygon is not None:
        poly = np.asarray(exclude_polygon, dtype=np.int32).reshape(-1, 2)
        if len(poly) >= 3:
            cv2.fillPoly(thresh, [poly], 0)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        h, w = mask.shape
        if 0.005 * h * w < area < 0.95 * h * w:
            clean = np.zeros_like(mask)
            cv2.drawContours(clean, [largest], -1, 255, cv2.FILLED)
            mask = clean

    return (mask > 127).astype(np.uint8)


def segment_lesion(
    image_path: str | Path,
    remove_hair_flag: bool = False,
    exclude_polygon=None,
    use_deep_model: bool = True,
) -> np.ndarray:
    """
    Main segmentation function.
    Uses trained UNet model if available; falls back to classical segmentation.
    Returns binary mask (values 0 or 1).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    if remove_hair_flag:
        image = remove_hair(image)

    h, w = image.shape[:2]
    unet = _get_unet_model() if use_deep_model else None

    if unet is not None:
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (256, 256))
            tensor_in = torch.from_numpy(resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

            with torch.no_grad():
                logits = unet(tensor_in)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            mask_256 = (prob > 0.5).astype(np.uint8)
            mask = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)

            if exclude_polygon is not None:
                poly = np.asarray(exclude_polygon, dtype=np.int32).reshape(-1, 2)
                if len(poly) >= 3:
                    cv2.fillPoly(mask, [poly], 0)

            # Keep largest connected component
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                clean = np.zeros_like(mask)
                cv2.drawContours(clean, [largest], -1, 1, cv2.FILLED)
                return clean

        except Exception as exc:
            print(f"[segmentation] UNet inference fallback: {exc}")

    # Fallback to classical segmentation
    return segment_lesion_classical(image, exclude_polygon=exclude_polygon)
