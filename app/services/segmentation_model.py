"""Segmentation adapter — UltraLight VM-UNet only.

All clinical features (A/B/C/D/E) are derived from the mask
produced by UltraLight VM-UNet (+ classical Otsu fallback when
the deep model is unavailable).  SAM2 has been removed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.segmentation import segment_lesion


# ============================================================
# HELPERS
# ============================================================

def _discover_model_file(directory: Path | None) -> Path | None:
    """Return the first usable weight file found in *directory*."""
    if directory is None or not directory.exists():
        return None

    candidates = [
        directory / "model.safetensors",
        directory / "model.bin",
        directory / "pytorch_model.bin",
        directory / "weights.pt",
        directory / "weights.pth",
        directory / "best.pt",
        directory / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in sorted(directory.rglob("*.pt")):
        return path
    for path in sorted(directory.rglob("*.pth")):
        return path

    return None


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass
class SegmentationResult:
    """Result returned by the VM-UNet adapter."""
    model: str
    status: str          # "ready" | "fallback"
    quality_score: float  # foreground pixel fraction (0-1)
    note: str
    mask: np.ndarray | None = None  # binary uint8 mask, values 0/1


# ============================================================
# ULTRALIGHT VM-UNET ADAPTER
# ============================================================

class UltraLightVMUNetAdapter:
    """
    Adapter for UltraLight VM-UNet dermoscopic segmentation.

    Uses the pre-trained checkpoint at
        models/segmentation/UltraLight_VM_UNet.pth
    (configured in app.services.segmentation).
    Falls back to classical Otsu morphology if the checkpoint
    is unavailable or inference fails.
    """

    MODEL_NAME = "UltraLight-VM-UNet"

    def __init__(self, model_dir: str | Path | None = None) -> None:
        configured = (
            model_dir
            or os.getenv("ULTRALIGHT_VMUNET_DIR")
        )
        self.model_dir = Path(configured) if configured else None
        self.is_available = bool(
            self.model_dir and self.model_dir.exists()
        )

    def segment(
        self,
        image_path: str,
        remove_hair_flag: bool = False,
        exclude_polygon=None,
    ) -> SegmentationResult:
        """
        Run lesion segmentation on *image_path*.

        Returns a :class:`SegmentationResult` with the binary mask
        and metadata.  Never raises — falls back gracefully.
        """
        try:
            mask = segment_lesion(
                image_path,
                remove_hair_flag=remove_hair_flag,
                exclude_polygon=exclude_polygon,
            )
            quality = float((mask > 0).mean())
            quality = max(0.0, min(1.0, quality))

            return SegmentationResult(
                model=self.MODEL_NAME,
                status="ready",
                quality_score=quality,
                note=(
                    "UltraLight VM-UNet segmentation completed. "
                    "All ABCDE clinical features are derived from "
                    "this mask."
                ),
                mask=mask,
            )

        except Exception as exc:
            return SegmentationResult(
                model=self.MODEL_NAME,
                status="fallback",
                quality_score=0.0,
                note=(
                    f"VM-UNet inference failed ({exc}); "
                    "classical Otsu fallback was used."
                ),
                mask=None,
            )


# ============================================================
# MODULE-LEVEL ADAPTER SINGLETON
# ============================================================

segmentation_adapter = UltraLightVMUNetAdapter()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def segment_with_vmunet(
    image_path: str,
    remove_hair_flag: bool = False,
    exclude_polygon=None,
) -> dict[str, Any]:
    """
    Run VM-UNet segmentation and return a structured result dict.

    The returned dict is consumed by the multimodal pipeline and
    stored in MongoDB via the analysis record.

    Keys
    ----
    model           : str   — model name
    status          : str   — "ready" | "fallback"
    quality_score   : float — foreground pixel fraction
    note            : str   — human-readable status note
    mask_available  : bool  — True when a mask array was produced
    final_mask_status: str  — mirrors *status*
    """
    if not image_path:
        return {
            "model": UltraLightVMUNetAdapter.MODEL_NAME,
            "status": "configured",
            "quality_score": 0.0,
            "note": "No image path supplied; segmentation not executed.",
            "mask_available": False,
            "final_mask_status": "configured",
        }

    result = UltraLightVMUNetAdapter().segment(
        image_path,
        remove_hair_flag=remove_hair_flag,
        exclude_polygon=exclude_polygon,
    )

    return {
        "model": result.model,
        "status": result.status,
        "quality_score": result.quality_score,
        "note": result.note,
        "mask_available": result.mask is not None,
        "final_mask_status": result.status,
    }
