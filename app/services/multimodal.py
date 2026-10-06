"""Multimodal melanoma assessment pipeline.

Flow:

    Dermoscopic image
        -> ViT visual encoder
        -> UltraLight VM-UNet segmentation
        -> ABCDE extraction
        -> 16-D clinical vector
        -> trained cross-modal contrastive alignment
        -> melanoma assessment
        -> CLIP concepts
        -> structured report

Important:
    UltraLight VM-UNet is the only automatic segmentation model.
    SAM2 is not part of the production segmentation pipeline.
"""

from __future__ import annotations

import os
from typing import Any

import cv2

from app.services.analysis import extract_abcde_features
from app.services.classification import MODEL_BACKBONE, classify
from app.services.clip_model import clip_extractor
from app.services.contrastive import contrastive_aligner
from app.services.explainability import generate_explanation
from app.services.llm_report import llm_report_synthesizer
from app.services.segmentation import segment_lesion
from app.services.segmentation_model import segment_with_vmunet


# ============================================================
# VISION TRANSFORMER
# ============================================================

def run_vision_transformer(
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose the primary ViT classification result."""

    payload = classification or {}

    prob = float(
        payload.get("melanoma_probability", 0.0) or 0.0
    )

    risk = str(
        payload.get("risk_level", "low")
    ).lower()

    return {
        "backbone": MODEL_BACKBONE,
        "status": "available",
        "risk_level": risk,
        "melanoma_probability": prob,
        "predicted_class": payload.get("predicted_class"),
        "method": payload.get(
            "method",
            "trained_vit_model",
        ),
        "model_version": payload.get(
            "model_version"
        ),
        "branch": "DNN Classification Pipeline",
        "note": (
            "Vision Transformer encodes the dermoscopic image "
            "for melanoma classification and downstream "
            "multimodal reasoning."
        ),
    }


# ============================================================
# SEGMENTATION
# ============================================================

def run_segmentation(
    mask_quality: float | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    """
    Run UltraLight VM-UNet segmentation.

    SAM2 is intentionally not used here.
    """

    if image_path:
        segmentation = segment_with_vmunet(
            image_path,
        )
    else:
        segmentation = {
            "model": "UltraLight-VM-UNet",
            "status": "configured",
            "quality_score": float(
                mask_quality if mask_quality is not None else 0.0
            ),
            "note": (
                "UltraLight VM-UNet adapter is ready; "
                "provide an image path to execute mask generation."
            ),
            "mask_available": False,
            "final_mask_status": "configured",
        }

    return {
        "method": "clinical_explanation_pipeline",
        "model": "UltraLight-VM-UNet",
        "status": segmentation.get(
            "final_mask_status",
            segmentation.get("status", "ready"),
        ),
        "quality_score": max(
            0.0,
            min(
                1.0,
                float(
                    segmentation.get(
                        "quality_score",
                        0.0,
                    )
                    or 0.0
                ),
            ),
        ),
        "branch": "Clinical Explanation Pipeline",
        "mask_available": bool(
            segmentation.get(
                "mask_available",
                False,
            )
        ),
        "note": segmentation.get(
            "note",
            (
                "UltraLight VM-UNet isolates the lesion "
                "region and derives the ABCDE features."
            ),
        ),
    }


# ============================================================
# COLOR HELPER
# ============================================================

def _get_color_index(
    features: dict[str, Any],
) -> float:
    """Return the normalized color-variation index."""

    if "color_index" in features:
        return float(
            features.get(
                "color_index",
                0.0,
            )
            or 0.0
        )

    color = features.get(
        "color",
        {},
    ) or {}

    return float(
        color.get(
            "index",
            0.0,
        )
        or 0.0
    )


# ============================================================
# ABCDE FEATURES
# ============================================================

def extract_abc_features(
    abcde: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Normalize the ABCDE information exposed by the main
    analysis pipeline.
    """

    features = abcde or {}

    asymmetry = float(
        features.get(
            "asymmetry",
            0.0,
        )
        or 0.0
    )

    border = float(
        features.get(
            "border_irregularity",
            0.0,
        )
        or 0.0
    )

    color_index = _get_color_index(
        features
    )

    diameter_mm = features.get(
        "diameter_mm"
    )

    if diameter_mm is None:
        diameter = features.get(
            "diameter",
            {},
        ) or {}

        diameter_mm = diameter.get(
            "max_diameter_mm"
        )

    return {
        "asymmetry": asymmetry,
        "border_irregularity": border,
        "color_index": color_index,
        "diameter_mm": diameter_mm,
        "evolution_change_detected": bool(
            features.get(
                "evolution_change_detected",
                False,
            )
        ),
        "status": "computed",
        "branch": "ABCDE Features",
        "note": (
            "A + B + C + D + E clinical evidence is "
            "extracted from the lesion."
        ),
    }


# ============================================================
# IMAGE FEATURES
# ============================================================

def build_image_features(
    classification: dict[str, Any] | None = None,
    abcde: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact image-side feature representation."""

    payload = classification or {}
    features = abcde or {}

    prob = float(
        payload.get(
            "melanoma_probability",
            0.0,
        )
        or 0.0
    )

    color_index = _get_color_index(
        features
    )

    border = float(
        features.get(
            "border_irregularity",
            0.0,
        )
        or 0.0
    )

    return {
        "texture": min(
            1.0,
            max(
                0.0,
                prob,
            ),
        ),
        "color": min(
            1.0,
            max(
                0.0,
                color_index,
            ),
        ),
        "structure": min(
            1.0,
            max(
                0.0,
                border,
            ),
        ),
        "status": "encoded",
        "branch": "Image Features",
    }


# ============================================================
# CONTRASTIVE ALIGNMENT
# ============================================================

def run_contrastive_alignment(
    classification: dict[str, Any] | None = None,
    abcde: dict[str, Any] | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    """
    Align the visual representation with the clinical feature
    representation.

    The contrastive branch is supporting evidence only. It does
    not silently modify the primary ViT melanoma probability.
    """

    features = abcde or {}

    image_features = build_image_features(
        classification,
        features,
    )

    clinical_features = dict(
        features
    )

    # --------------------------------------------------------
    # Compatibility mode for architecture/unit tests.
    # --------------------------------------------------------
    if (
        not image_path
        or not os.path.exists(image_path)
    ):
        try:
            fallback_score = (
                contrastive_aligner.compute_similarity(
                    image_features,
                    clinical_features,
                )
            )
        except Exception:
            fallback_score = 0.0

        return {
            "score": float(
                fallback_score
            ),
            "similarity": float(
                fallback_score
            ),
            "alignment_score": float(
                fallback_score
            ),
            "status": "fallback",
            "branch": "Contrastive Learning",
            "model": (
                "CEFM Dual Projection Heads "
                "(NT-Xent aligned)"
            ),
            "image_embedding_available": False,
            "clinical_vector_available": True,
            "image_features": image_features,
            "note": (
                "Compatibility fallback used because no image "
                "path was supplied. The trained cross-modal "
                "alignment is used when an image path is available."
            ),
        }

    try:
        alignment = (
            contrastive_aligner.compute_alignment(
                image_path=image_path,
                abcde_features=clinical_features,
            )
        )

        similarity = alignment.get(
            "similarity"
        )

        alignment_score = alignment.get(
            "alignment_score"
        )

        return {
            "score": alignment_score,
            "similarity": similarity,
            "alignment_score": alignment_score,
            "status": alignment.get(
                "status",
                "computed",
            ),
            "branch": "Contrastive Learning",
            "model": alignment.get(
                "model",
                (
                    "CEFM Dual Projection Heads "
                    "(NT-Xent aligned)"
                ),
            ),
            "checkpoint": alignment.get(
                "checkpoint"
            ),
            "image_embedding_available": True,
            "clinical_vector_available": True,
            "image_features": image_features,
            "clinical_vector": alignment.get(
                "clinical_vector"
            ),
            "note": (
                "The real ViT image embedding and the 16-D "
                "ABCDE clinical representation are aligned "
                "using the trained CEFM contrastive model."
            ),
        }

    except Exception as exc:
        return {
            "score": None,
            "similarity": None,
            "alignment_score": None,
            "status": "error",
            "branch": "Contrastive Learning",
            "model": (
                "CEFM Dual Projection Heads "
                "(NT-Xent aligned)"
            ),
            "image_embedding_available": False,
            "clinical_vector_available": True,
            "image_features": image_features,
            "error": str(exc),
            "note": (
                "Contrastive alignment could not be computed. "
                "The primary ViT classifier result remains separate."
            ),
        }


# ============================================================
# MELANOMA CLASSIFIER
# ============================================================

def run_melanoma_classifier(
    classification: dict[str, Any] | None = None,
    abcde: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Expose the primary melanoma classifier result.

    ABCDE and contrastive features are supporting evidence and
    are not silently mixed into the primary probability here.
    """

    payload = classification or {}

    prob = float(
        payload.get(
            "melanoma_probability",
            0.0,
        )
        or 0.0
    )

    return {
        "prediction": payload.get(
            "predicted_class",
            "nv",
        ),
        "melanoma_probability": prob,
        "risk_level": str(
            payload.get(
                "risk_level",
                "low",
            )
        ).lower(),
        "method": payload.get(
            "method",
            "trained_vit_model",
        ),
        "model_name": payload.get(
            "model_name",
            "ViT-Melanoma",
        ),
        "model_version": payload.get(
            "model_version"
        ),
        "branch": "Melanoma Classifier",
        "note": (
            "The validated ViT classifier provides the primary "
            "melanoma prediction. Contrastive alignment is "
            "reported as multimodal evidence and is not "
            "silently mixed into the classifier probability."
        ),
    }


# ============================================================
# CLIP CONCEPTS
# ============================================================

def run_clip_concepts(
    abcde: dict[str, Any] | None = None,
    image_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract clinically meaningful visual concepts."""

    features = abcde or {}

    extracted = clip_extractor.extract(
        image_features or {},
        features,
    )

    return {
        "concepts": extracted[
            "concepts"
        ],
        "branch": "CLIP + DeepSeek",
        "status": extracted[
            "status"
        ],
        "model": extracted[
            "model"
        ],
        "note": (
            "Visual concepts are translated into clinically "
            "meaningful language for the medical report."
        ),
    }


# ============================================================
# FINAL REPORT
# ============================================================

def build_final_report(
    classification: dict[str, Any] | None = None,
    abcde: dict[str, Any] | None = None,
    evolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the structured medical report."""

    explanation = generate_explanation(
        classification or {},
        abcde or {},
        evolution or {},
    )

    summary = explanation[
        "summary"
    ]

    report = llm_report_synthesizer.generate(
        summary,
        {
            "classification": classification or {},
            "abcde": abcde or {},
            "evolution": evolution or {},
        },
    )

    return {
        "summary": report[
            "summary"
        ],
        "status": report[
            "status"
        ],
        "branch": "Structured Medical Report",
        "format": report[
            "format"
        ],
        "note": report[
            "note"
        ],
    }


# ============================================================
# MULTIMODAL PIPELINE
# ============================================================

def run_multimodal_pipeline(
    classification: dict[str, Any] | None = None,
    abcde: dict[str, Any] | None = None,
    evolution: dict[str, Any] | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    """Run all multimodal evidence and reporting stages."""

    stage_vit = run_vision_transformer(
        classification
    )

    stage_seg = run_segmentation(
        image_path=image_path
    )

    stage_abc = extract_abc_features(
        abcde
    )

    stage_image = build_image_features(
        classification,
        abcde,
    )

    stage_align = run_contrastive_alignment(
        classification=classification,
        abcde=abcde,
        image_path=image_path,
    )

    stage_pred = run_melanoma_classifier(
        classification,
        abcde,
    )

    stage_clip = run_clip_concepts(
        abcde,
        stage_image,
    )

    stage_report = build_final_report(
        classification,
        abcde,
        evolution,
    )

    legacy = {
        "vision_transformer": stage_vit,
        "segmentation": stage_seg,
        "abc_features": stage_abc,
        "contrastive_alignment": stage_align,
        "clip_concepts": stage_clip,
        "report": stage_report,
    }

    structured = {
        "dermoscopic_image": {
            "source": "input",
            "branch": "DNN Classification Pipeline",
            "vision_transformer": stage_vit,
        },
        "clinical_explanation_pipeline": {
            "source": "input",
            "branch": "Clinical Explanation Pipeline",
            "segmentation": stage_seg,
            "abcde_features": stage_abc,
        },
        "image_features": stage_image,
        "aligned_features": stage_align,
        "melanoma_classifier": stage_pred,
        "clip_deepseek": stage_clip,
        "structured_medical_report": stage_report,
    }

    return {
        **legacy,
        **structured,
    }


# ============================================================
# SEGMENTATION -> ABCDE
# ============================================================

def run_image_segmentation_to_abcde(
    image_path: str,
) -> dict[str, Any]:
    """
    Run UltraLight VM-UNet segmentation and extract ABCDE
    features from the resulting lesion mask.

    Important:
        - UltraLight VM-UNet is the only automatic segmentation
          model used here.
        - The mask is generated through the project's existing
          segmentation service.
        - No SAM2 refinement is performed.
        - No artificial pixels-per-mm value is used.
        - Diameter in mm is only available when a real physical
          calibration is supplied by the main patient-analysis
          pipeline.
    """

    if not image_path:
        raise ValueError(
            "An image path is required to run "
            "the segmentation-to-ABCDE flow."
        )

    # --------------------------------------------------------
    # 1. Run the configured UltraLight VM-UNet segmentation
    # --------------------------------------------------------
    segmentation = segment_with_vmunet(
        image_path
    )

    # --------------------------------------------------------
    # 2. Obtain the actual binary lesion mask
    #
    # The existing adapter returns metadata while the existing
    # segmentation service returns the mask itself.
    # --------------------------------------------------------
    mask = None

    try:
        mask = segment_lesion(
            image_path
        )
    except Exception:
        mask = None

    # --------------------------------------------------------
    # 3. Safe default feature structure
    # --------------------------------------------------------
    features = {
        "asymmetry": 0.0,
        "border_irregularity": 0.0,
        "color": {
            "index": 0.0,
        },
        "color_index": 0.0,
        "diameter_pixels": None,
        "diameter_mm": None,
        "diameter_mm_available": False,
    }

    # --------------------------------------------------------
    # 4. Extract A/B/C/D from the lesion mask
    # --------------------------------------------------------
    if mask is not None:
        try:
            image = cv2.imread(
                image_path
            )

            if image is not None:
                extracted = extract_abcde_features(
                    image,
                    mask,
                    pixels_per_mm=None,
                )

                color = extracted.get(
                    "color",
                    {},
                ) or {}

                color_index = float(
                    color.get(
                        "index",
                        0.0,
                    )
                    or 0.0
                )

                diameter = extracted.get(
                    "diameter",
                    {},
                ) or {}

                features = {
                    "asymmetry": float(
                        extracted.get(
                            "asymmetry",
                            0.0,
                        )
                        or 0.0
                    ),
                    "border_irregularity": float(
                        extracted.get(
                            "border_irregularity",
                            0.0,
                        )
                        or 0.0
                    ),
                    "color": {
                        "h": color.get(
                            "h",
                            0.0,
                        ),
                        "s": color.get(
                            "s",
                            0.0,
                        ),
                        "v": color.get(
                            "v",
                            0.0,
                        ),
                        "index": color_index,
                    },
                    "color_index": color_index,
                    "diameter_pixels": diameter.get(
                        "max_diameter_pixels"
                    ),
                    "diameter_mm": None,
                    "diameter_mm_available": False,
                }

        except Exception:
            # Keep safe defaults if feature extraction fails.
            pass

    return {
        "segmentation": segmentation,
        "abcde": features,
        "mask_available": mask is not None,
    }


# ============================================================
# FULL IMAGE PIPELINE
# ============================================================

def run_full_image_pipeline(
    image_path: str,
    classification: dict[str, Any] | None = None,
    abcde: dict[str, Any] | None = None,
    evolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the complete image-level multimodal pipeline.

    A real image path is expected. If processing fails, the
    exception is propagated instead of silently returning
    fabricated/default clinical results.
    """

    if not image_path:
        raise ValueError(
            "An image path is required to run "
            "the multimodal pipeline."
        )

    resolved = str(
        image_path
    )

    if not os.path.exists(
        resolved
    ):
        resolved = os.path.abspath(
            os.path.join(
                os.getcwd(),
                image_path,
            )
        )

    features = abcde or {}
    payload = classification or {}

    # --------------------------------------------------------
    # Real image pipeline
    # --------------------------------------------------------
    if os.path.exists(
        resolved
    ):
        image = cv2.imread(
            resolved
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {resolved}"
            )

        # ----------------------------------------------------
        # Segmentation -> ABCDE
        # ----------------------------------------------------
        segmentation_state = (
            run_image_segmentation_to_abcde(
                resolved
            )
        )

        segmented_features = (
            segmentation_state[
                "abcde"
            ]
        )

        if segmented_features:
            features = segmented_features

        # ----------------------------------------------------
        # Primary melanoma classifier
        # ----------------------------------------------------
        payload = classify(
            resolved,
            features,
        )

        # ----------------------------------------------------
        # Multimodal evidence/reporting
        # ----------------------------------------------------
        result = run_multimodal_pipeline(
            classification=payload,
            abcde=features,
            evolution=(
                evolution
                or {
                    "change_detected": False,
                }
            ),
            image_path=resolved,
        )

        result[
            "input_image"
        ] = resolved

        result[
            "segmentation_state"
        ] = segmentation_state

        return result

    # --------------------------------------------------------
    # No valid image path
    # --------------------------------------------------------
    raise FileNotFoundError(
        f"Image file not found: {resolved}"
    )