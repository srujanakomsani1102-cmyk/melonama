import json
import shutil
import uuid
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response

from app.core.config import DATA_DIR
from app.database import (
    cases_collection,
    analyses_collection,
)
from app.services.analysis import analyze_image
from app.services.visualization import render_view


# ============================================================
# UPLOAD STORAGE
# ============================================================

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


router = APIRouter()


ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


# ============================================================
# MONGODB INDEXES
# ============================================================

try:
    cases_collection.create_index(
        [("label", 1), ("created_at", -1)]
    )

    analyses_collection.create_index(
        [("case_id", 1), ("created_at", -1)]
    )

except Exception as exc:
    print(f"[MongoDB WARNING] Could not create indexes: {exc}")


# ============================================================
# CASE HELPERS
# ============================================================

def _resolve_patient_case(
    case_id: str | None = None,
    patient_name: str | None = None,
):
    """
    Return an existing MongoDB case or create a new one.

    The case label acts as the patient/case identifier so that
    multiple uploaded images can be grouped together for
    Evolution (E) tracking.
    """

    # --------------------------------------------------------
    # Existing case requested
    # --------------------------------------------------------

    if case_id:

        case = cases_collection.find_one(
            {"_id": case_id}
        )

        if not case:
            raise HTTPException(
                status_code=404,
                detail=f"Case {case_id} not found.",
            )

        return case


    # --------------------------------------------------------
    # Resolve case label
    # --------------------------------------------------------

    patient_label = (
        patient_name or ""
    ).strip()

    if not patient_label:
        patient_label = (
            f"Case {datetime.utcnow():%Y-%m-%d %H:%M}"
        )


    # --------------------------------------------------------
    # Find most recent case with same label
    # --------------------------------------------------------

    case = cases_collection.find_one(
        {"label": patient_label},
        sort=[
            ("created_at", -1)
        ],
    )


    if case is not None:
        return case


    # --------------------------------------------------------
    # Create new case
    # --------------------------------------------------------

    case_id = uuid.uuid4().hex

    case = {
        "_id": case_id,
        "label": patient_label,
        "created_at": datetime.utcnow(),
    }

    cases_collection.insert_one(case)

    return case


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health_check():

    return {
        "status": "ok",
        "message": "CEFM Backend is running",
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    case_id: str | None = Form(None),
    case_label: str | None = Form(None),
    patient_name: str | None = Form(None),
    remove_hair: bool = Form(False),
    pixels_per_mm: float | None = Form(None),
):
    """
    Upload an image and run the full melanoma pipeline.

    Pipeline:

    validation
        -> image type
        -> calibration
        -> segmentation
        -> ABCDE
        -> classification
        -> evolution
        -> RAG
        -> report

    Image metadata and analysis results are stored in MongoDB.
    """


    # ========================================================
    # 1. VALIDATE FILE TYPE
    # ========================================================

    if image.content_type not in ALLOWED_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG images "
                "are supported."
            ),
        )


    # ========================================================
    # 2. SAVE IMAGE
    # ========================================================

    file_id = uuid.uuid4().hex

    extension = (
        Path(
            image.filename or "image.jpg"
        ).suffix.lower()
        or ".jpg"
    )

    stored_path = (
        UPLOAD_DIR /
        f"{file_id}{extension}"
    )


    try:

        with open(
            stored_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                image.file,
                buffer,
            )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not save image: {exc}",
        )


    # ========================================================
    # 3. RESOLVE CASE
    # ========================================================

    resolved_patient_name = (
        patient_name
        or case_label
        or ""
    ).strip()


    try:

        case = _resolve_patient_case(
            case_id=case_id,
            patient_name=resolved_patient_name,
        )

    except Exception:

        stored_path.unlink(
            missing_ok=True
        )

        raise


    resolved_case_id = case["_id"]


    # ========================================================
    # 4. GET PREVIOUS ANALYSIS FOR EVOLUTION
    # ========================================================

    previous = analyses_collection.find_one(
        {
            "case_id": resolved_case_id,
            "is_valid": True,
        },
        sort=[
            ("created_at", -1)
        ],
    )


    previous_features = None


    if previous:

        previous_features = {
            "asymmetry": previous.get(
                "asymmetry"
            ),

            "border_irregularity": previous.get(
                "border_irregularity"
            ),

            "color_h": previous.get(
                "color_h"
            ),

            "color_s": previous.get(
                "color_s"
            ),

            "color_v": previous.get(
                "color_v"
            ),

            "color_index": previous.get(
                "color_index"
            ),

            "diameter_pixels": previous.get(
                "diameter_pixels"
            ),

            "diameter_mm": previous.get(
                "diameter_mm"
            ),
        }


    # ========================================================
    # 5. RUN COMPLETE MELANOMA ANALYSIS
    # ========================================================

    try:

        result = analyze_image(
            str(stored_path),
            image.filename or "",
            previous_features,
            remove_hair_flag=remove_hair,
            pixels_per_mm=pixels_per_mm,
        )

    except Exception as exc:

        # Print the complete traceback in the terminal for debugging.
        traceback.print_exc()

        stored_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )


    # ========================================================
    # 6. HANDLE INVALID IMAGE
    # ========================================================

    if result["status"] == "invalid":

        stored_path.unlink(
            missing_ok=True
        )

        validation = result.get(
            "validation",
            {},
        )

        if not isinstance(
            validation,
            dict,
        ):
            validation = {}


        reason = None


        # Calibration rejection
        calibration_rejection = validation.get(
            "calibration_rejection",
            {},
        )

        if isinstance(
            calibration_rejection,
            dict,
        ):

            reason = (
                calibration_rejection.get(
                    "message"
                )
                or reason
            )


        # Post segmentation quality
        if not reason:

            post_quality = validation.get(
                "post_segmentation_quality",
                {},
            )

            if isinstance(
                post_quality,
                dict,
            ):

                reason = (
                    post_quality.get(
                        "reason"
                    )
                    or reason
                )


        # Diameter rejection
        if not reason:

            diameter_rejection = validation.get(
                "diameter_rejection",
                {},
            )

            if isinstance(
                diameter_rejection,
                dict,
            ):

                reason = (
                    diameter_rejection.get(
                        "message"
                    )
                    or reason
                )


        # Basic validation
        if not reason:

            basic_reason = validation.get(
                "reason"
            )

            if (
                basic_reason
                and basic_reason
                != "All basic image quality checks passed."
            ):

                reason = basic_reason


        if not reason:

            reason = (
                "The uploaded image did not "
                "satisfy the required quality "
                "or calibration checks."
            )


        if isinstance(
            reason,
            list,
        ):

            reason = " ".join(
                str(item)
                for item in reason
            )


        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Image validation failed."
                ),
                "reason": str(reason),
                "validation": validation,
                "rejection_stage": result.get(
                    "rejection_stage"
                ),
                "segmentation_url": (
                    f"/api/v1/segmentation/{file_id}"
                    if result.get("segmentation_path")
                    else None
                ),
            },
        )


    # ========================================================
    # 7. EXTRACT RESULTS
    # ========================================================

    features = result["features"]

    classification = result[
        "classification"
    ]

    evolution = result[
        "evolution"
    ]

    response_evolution = (
        evolution
        if previous is not None
        else None
    )


    # ========================================================
    # 8. CREATE MONGODB ANALYSIS DOCUMENT
    # ========================================================

    analysis_id = uuid.uuid4().hex

    created_at = datetime.utcnow()


    record = {

        "_id": analysis_id,

        "case_id": resolved_case_id,

        "original_filename": (
            image.filename
        ),

        "stored_path": str(
            stored_path
        ),

        "segmentation_path": result.get(
            "segmentation_path"
        ),

        "is_valid": True,


        # ----------------------------------------------------
        # Image information
        # ----------------------------------------------------

        "image_type": result.get(
            "image_type",
            {},
        ),


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        "predicted_class": classification.get(
            "predicted_class"
        ),

        "melanoma_probability": classification.get(
            "melanoma_probability"
        ),

        "risk_level": classification.get(
            "risk_level"
        ),


        # ----------------------------------------------------
        # ABCDE
        # ----------------------------------------------------

        "asymmetry": features.get(
            "asymmetry"
        ),

        "border_irregularity": features.get(
            "border_irregularity"
        ),

        "color_h": features.get(
            "color_h"
        ),

        "color_s": features.get(
            "color_s"
        ),

        "color_v": features.get(
            "color_v"
        ),

        "color_index": features.get(
            "color_index"
        ),

        "diameter_pixels": features.get(
            "diameter_pixels"
        ),

        "diameter_mm": features.get(
            "diameter_mm"
        ),


        # ----------------------------------------------------
        # Evolution
        # ----------------------------------------------------

        "evolution_available": evolution.get(
            "available",
            False,
        ),

        "evolution_change_detected": evolution.get(
            "change_detected",
            False,
        ),

        "evolution": evolution,

        # ----------------------------------------------------
        # Multimodal / Contrastive Alignment
        # ----------------------------------------------------

        "multimodal": result.get(
            "multimodal",
            {},
        ),

        "contrastive_alignment": result.get(
            "contrastive_alignment",
            {},
        ),

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        "rag_evidence": result.get(
            "rag_evidence",
            {},
        ),


        # ----------------------------------------------------
        # Explanation
        # ----------------------------------------------------

        "explanation": result.get(
            "explanation",
            {},
        ),


        # ----------------------------------------------------
        # Complete report
        # ----------------------------------------------------

        "report": result.get(
            "report",
            {},
        ),


        "created_at": created_at,
    }


    # ========================================================
    # 9. SAVE ANALYSIS TO MONGODB
    # ========================================================

    try:

        analyses_collection.insert_one(
            record
        )

    except Exception as exc:

        stored_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save analysis "
                f"to MongoDB: {exc}"
            ),
        )


    # ========================================================
    # 10. RESPONSE
    # ========================================================

    return {

        "status": "success",

        "analysis_id": analysis_id,

        "case_id": resolved_case_id,

        "image_url": (
            f"/api/v1/images/{analysis_id}"
        ),

        "segmentation_url": (
            f"/api/v1/images/{analysis_id}/segmentation"
            if result.get("segmentation_path")
            else None
        ),

        "filename": image.filename,

        "validation": result[
            "validation"
        ],

        "image_type": result[
            "image_type"
        ],

        "calibration": result.get(
            "calibration"
        ),

        "abcde": {
            "asymmetry": features.get("asymmetry"),
            "border_irregularity": features.get("border_irregularity"),
            "color_h": features.get("color_h"),
            "color_s": features.get("color_s"),
            "color_v": features.get("color_v"),
            "color_index": features.get("color_index"),
            "diameter_pixels": features.get("diameter_pixels"),
            "diameter_mm": features.get("diameter_mm"),
        },

        "classification": classification,

        "evolution": response_evolution,

        "multimodal": result.get(
            "multimodal",
            {},
        ),

        "contrastive_alignment": result.get(
            "contrastive_alignment",
            {},
        ),

        "explanation": result[
            "explanation"
        ],

        "rag_evidence": result.get(
            "rag_evidence",
            {},
        ),

        "report": result[
            "report"
        ],
    }


# ============================================================
# CREATE CASE
# ============================================================

@router.post("/cases")
def create_case(
    label: str | None = Form(None),
):

    case_id = uuid.uuid4().hex

    case_label = (
        label.strip()
        if label
        else None
    )


    if not case_label:

        case_label = (
            f"Case {datetime.utcnow():%Y-%m-%d %H:%M}"
        )


    case = {

        "_id": case_id,

        "label": case_label,

        "created_at": datetime.utcnow(),
    }


    cases_collection.insert_one(
        case
    )


    return {

        "case_id": case_id,

        "label": case_label,
    }


# ============================================================
# LIST CASES
# ============================================================

@router.get("/cases")
def list_cases():

    cases = cases_collection.find(
        {}
    ).sort(
        "created_at",
        -1,
    )


    response = []


    for case in cases:

        number_of_analyses = (
            analyses_collection.count_documents(
                {
                    "case_id": case["_id"]
                }
            )
        )


        created_at = case.get(
            "created_at"
        )


        response.append(
            {

                "case_id": case["_id"],

                "label": case.get(
                    "label"
                ),

                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),

                "num_analyses": (
                    number_of_analyses
                ),
            }
        )


    return response


# ============================================================
# GET SINGLE CASE
# ============================================================

@router.get("/cases/{case_id}")
def get_case(
    case_id: str,
):

    case = cases_collection.find_one(
        {
            "_id": case_id
        }
    )


    if not case:

        raise HTTPException(
            status_code=404,
            detail="Case not found.",
        )


    analyses = analyses_collection.find(
        {
            "case_id": case_id
        }
    ).sort(
        "created_at",
        1,
    )


    analysis_response = []


    for analysis in analyses:

        created_at = analysis.get(
            "created_at"
        )


        analysis_response.append(
            {

                "analysis_id": analysis["_id"],

                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),

                "predicted_class": analysis.get(
                    "predicted_class"
                ),

                "melanoma_probability": analysis.get(
                    "melanoma_probability"
                ),

                "risk_level": analysis.get(
                    "risk_level"
                ),

                "diameter_pixels": analysis.get(
                    "diameter_pixels"
                ),

                "diameter_mm": analysis.get(
                    "diameter_mm"
                ),

                "asymmetry": analysis.get(
                    "asymmetry"
                ),

                "border_irregularity": analysis.get(
                    "border_irregularity"
                ),

                "evolution_change_detected": analysis.get(
                    "evolution_change_detected",
                    False,
                ),

                "image_url": (
                    f"/api/v1/images/"
                    f"{analysis['_id']}"
                ),
            }
        )


    created_at = case.get(
        "created_at"
    )


    return {

        "case_id": case["_id"],

        "label": case.get(
            "label"
        ),

        "created_at": (
            created_at.isoformat()
            if created_at
            else None
        ),

        "analyses": analysis_response,
    }


# ============================================================
# LIST ANALYSES
# ============================================================

@router.get("/analyses")
def list_analyses(
    limit: int = 50,
):

    # Prevent unreasonable requests
    limit = max(
        1,
        min(limit, 500),
    )


    rows = analyses_collection.find(
        {}
    ).sort(
        "created_at",
        -1,
    ).limit(
        limit
    )


    response = []


    for analysis in rows:

        created_at = analysis.get(
            "created_at"
        )


        response.append(
            {

                "analysis_id": analysis["_id"],

                "case_id": analysis.get(
                    "case_id"
                ),

                "filename": analysis.get(
                    "original_filename"
                ),

                "created_at": (
                    created_at.isoformat()
                    if created_at
                    else None
                ),

                "predicted_class": analysis.get(
                    "predicted_class"
                ),

                "melanoma_probability": analysis.get(
                    "melanoma_probability"
                ),

                "risk_level": analysis.get(
                    "risk_level"
                ),

                "diameter_pixels": analysis.get(
                    "diameter_pixels"
                ),

                "diameter_mm": analysis.get(
                    "diameter_mm"
                ),

                "image_url": (
                    f"/api/v1/images/"
                    f"{analysis['_id']}"
                ),
            }
        )


    return response


# ============================================================
# GET SINGLE ANALYSIS
# ============================================================

@router.get("/analyses/{analysis_id}")
def get_analysis(
    analysis_id: str,
):

    analysis = analyses_collection.find_one(
        {
            "_id": analysis_id
        }
    )


    if not analysis:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )


    created_at = analysis.get(
        "created_at"
    )


    return {

        "analysis_id": analysis["_id"],

        "case_id": analysis.get(
            "case_id"
        ),

        "filename": analysis.get(
            "original_filename"
        ),

        "created_at": (
            created_at.isoformat()
            if created_at
            else None
        ),

        "image_url": (
            f"/api/v1/images/"
            f"{analysis['_id']}"
        ),

        "abcde": {

            "asymmetry": analysis.get(
                "asymmetry"
            ),

            "border_irregularity": analysis.get(
                "border_irregularity"
            ),

            "color_h": analysis.get(
                "color_h"
            ),

            "color_s": analysis.get(
                "color_s"
            ),

            "color_v": analysis.get(
                "color_v"
            ),

            "color_index": analysis.get(
                "color_index"
            ),

            "diameter_pixels": analysis.get(
                "diameter_pixels"
            ),

            "diameter_mm": analysis.get(
                "diameter_mm"
            ),
        },


        "classification": {

            "predicted_class": analysis.get(
                "predicted_class"
            ),

            "melanoma_probability": analysis.get(
                "melanoma_probability"
            ),

            "risk_level": analysis.get(
                "risk_level"
            ),
        },


        "evolution": analysis.get(
            "evolution"
        ),

        "multimodal": analysis.get(
            "multimodal",
            {},
        ),

        "contrastive_alignment": analysis.get(
            "contrastive_alignment",
            {},
        ),

        "rag_evidence": analysis.get(
            "rag_evidence",
            {},
        ),


        "explanation": analysis.get(
            "explanation",
            {},
        ),


        "report": analysis.get(
            "report"
        ),
    }


# ============================================================
# GET STORED IMAGE
# ============================================================

@router.get("/images/{analysis_id}")
def get_image(
    analysis_id: str,
):

    analysis = analyses_collection.find_one(
        {
            "_id": analysis_id
        }
    )


    if not analysis:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )


    stored_path = analysis.get(
        "stored_path"
    )


    if (
        not stored_path
        or not Path(stored_path).exists()
    ):

        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )


    return FileResponse(
        stored_path
    )


# ============================================================
# SEGMENTATION OVERLAY IMAGE
# ============================================================

@router.get("/images/{analysis_id}/segmentation")
def get_segmentation_image(
    analysis_id: str,
):
    """
    Serve the segmentation overlay PNG for an analysis.
    The overlay shows the lesion mask (red fill + cyan contour)
    on the original image. Available for ALL analyses, including
    those rejected by calibration or quality gates.
    """

    analysis = analyses_collection.find_one(
        {"_id": analysis_id}
    )

    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    seg_path = analysis.get("segmentation_path")

    if not seg_path or not Path(seg_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Segmentation image not available.",
        )

    return FileResponse(
        seg_path,
        media_type="image/png",
        filename=f"segmentation_{analysis_id}.png",
    )


# ============================================================
# XAI VISUAL EVIDENCE VIEWS
# ============================================================

@router.get(
    "/analyses/{analysis_id}/view/{view_name}"
)
def get_analysis_view(
    analysis_id: str,
    view_name: str,
    scale_px_per_mm: float | None = None,
):

    analysis = analyses_collection.find_one(
        {
            "_id": analysis_id
        }
    )


    if not analysis:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )


    stored_path = analysis.get(
        "stored_path"
    )


    if (
        not stored_path
        or not Path(stored_path).exists()
    ):

        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )


    try:

        png = render_view(
            view_name,
            stored_path,
            scale_px_per_mm,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate "
                f"visualization: {exc}"
            ),
        )


    return Response(
        content=png,
        media_type="image/png",
    )