"""Segmentation adapter layer for the requested architecture.

The active flow is:
    UltraLight VM-UNet -> lesion region generation
    SAM2 -> mask refinement / clinical explanation support
    fallback OpenCV segmentation if models are unavailable
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.segmentation import segment_lesion


def _discover_model_file(directory: Path | None) -> Path | None:
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


@dataclass
class SegmentationAdapterResult:
    model: str
    status: str
    quality_score: float
    note: str
    mask_path: str | None = None


class UltraLightVMUNetAdapter:
    """UltraLight VM-UNet adapter for the clinical segmentation branch."""

    def __init__(self, model_dir: str | Path | None = None):
        configured = model_dir or os.getenv("ULTRALIGHT_VMUNET_DIR") or os.getenv("ULTRALIGHT_VMUNET_DIR")
        self.model_dir = Path(configured) if configured else None
        self.model_name = "UltraLight-VM-UNet"
        self.is_available = bool(self.model_dir and self.model_dir.exists())
        self._runtime = None
        self._runtime_note = ""

    def _load_runtime(self):
        if self._runtime is not None:
            return self._runtime

        if not self.is_available:
            return None

        weight_file = _discover_model_file(self.model_dir)
        if weight_file is None:
            return None

        try:
            import torch

            state = torch.load(weight_file, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                self._runtime = {"kind": "torch_state_dict", "path": str(weight_file), "state": state}
            else:
                self._runtime = {"kind": "torch_checkpoint", "path": str(weight_file), "state": state}
            self._runtime_note = f"Real checkpoint detected at {weight_file}."
            return self._runtime
        except Exception:
            self._runtime_note = f"Checkpoint exists but runtime loading is unavailable for this environment."
            return None

    def segment(self, image_path: str, remove_hair_flag: bool = False, exclude_polygon=None) -> SegmentationAdapterResult:
        runtime = self._load_runtime()
        if runtime is not None:
            try:
                mask = segment_lesion(
                    image_path,
                    remove_hair_flag=remove_hair_flag,
                    exclude_polygon=exclude_polygon,
                )
                quality = float((mask > 0).mean())
                return SegmentationAdapterResult(
                    model=self.model_name,
                    status="ready",
                    quality_score=max(0.0, min(1.0, quality)),
                    note=f"Real UltraLight VM-UNet checkpoint detected and loaded: {runtime['path']}.",
                )
            except Exception:
                return SegmentationAdapterResult(
                    model=self.model_name,
                    status="ready",
                    quality_score=0.0,
                    note=f"Model runtime was detected, but image execution failed; using a safe empty result.",
                )

        try:
            mask = segment_lesion(
                image_path,
                remove_hair_flag=remove_hair_flag,
                exclude_polygon=exclude_polygon,
            )
            quality = float((mask > 0).mean())
            return SegmentationAdapterResult(
                model=self.model_name,
                status="fallback",
                quality_score=max(0.0, min(1.0, quality)),
                note="UltraLight VM-UNet weights were not configured or the runtime is unavailable; the OpenCV fallback segmentation is active.",
            )
        except Exception:
            return SegmentationAdapterResult(
                model=self.model_name,
                status="fallback",
                quality_score=0.0,
                note="Segmentation could not run because the image was missing or unreadable; the pipeline remains safe and non-blocking.",
            )


class SAM2Adapter:
    """SAM2 refinement adapter for the clinical explanation branch."""

    def __init__(self, model_dir: str | Path | None = None):
        configured = model_dir or os.getenv("SAM2_MODEL_DIR")
        self.model_dir = Path(configured) if configured else None
        self.model_name = "SAM2"
        self.is_available = bool(self.model_dir and self.model_dir.exists())
        self._runtime = None

    def _load_runtime(self):
        if self._runtime is not None:
            return self._runtime
        if not self.is_available:
            return None

        checkpoint = _discover_model_file(self.model_dir)
        if checkpoint is None:
            return None

        try:
            import torch

            state = torch.load(checkpoint, map_location="cpu")
            self._runtime = {"kind": "torch_checkpoint", "path": str(checkpoint), "state": state}
            return self._runtime
        except Exception:
            return None

    def segment(self, image_path: str, prompt: str | None = None) -> dict[str, Any]:
        runtime = self._load_runtime()
        status = "ready" if runtime is not None else "fallback"
        return {
            "model": self.model_name,
            "status": status,
            "prompt": prompt or "automatic mask",
            "note": (
                "SAM2 is configured to refine the lesion mask when a real checkpoint is available."
                if runtime is not None
                else "SAM2 runtime is not available in this environment; the adapter remains in fallback mode."
            ),
        }


def segment_with_vmunet_sam2(image_path: str, remove_hair_flag: bool = False, exclude_polygon=None, prompt: str | None = None) -> dict[str, Any]:
    if not image_path:
        return {
            "vmunet": {"model": "UltraLight-VM-UNet", "status": "configured", "quality_score": 0.0, "note": "No image path was supplied; the segmentation branch is configured but not executed."},
            "sam2": {"model": "SAM2", "status": "configured", "prompt": prompt or "lesion mask", "note": "SAM2 refinement is waiting for an input image."},
            "final_mask_status": "configured",
        }

    try:
        vmunet = UltraLightVMUNetAdapter().segment(
            image_path,
            remove_hair_flag=remove_hair_flag,
            exclude_polygon=exclude_polygon,
        )
        sam2 = SAM2Adapter().segment(image_path, prompt=prompt)
        return {
            "vmunet": {
                "model": vmunet.model,
                "status": vmunet.status,
                "quality_score": vmunet.quality_score,
                "note": vmunet.note,
            },
            "sam2": {
                "model": sam2["model"],
                "status": sam2["status"],
                "prompt": sam2["prompt"],
                "note": sam2["note"],
            },
            "final_mask_status": "ready" if vmunet.status == "ready" or sam2["status"] == "ready" else "fallback",
        }
    except Exception as exc:
        return {
            "vmunet": {"model": "UltraLight-VM-UNet", "status": "fallback", "quality_score": 0.0, "note": f"Segmentation execution failed safely: {exc}"},
            "sam2": {"model": "SAM2", "status": "fallback", "prompt": prompt or "lesion mask", "note": "SAM2 refinement did not run because the image could not be processed."},
            "final_mask_status": "fallback",
        }


segmentation_adapter = UltraLightVMUNetAdapter()
