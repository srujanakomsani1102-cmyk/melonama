# ============================================================
# MONGODB
# ============================================================

import os
from pathlib import Path

MONGODB_URL = "mongodb://localhost:27017"
MONGODB_DATABASE = "cefm_melanoma"


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================
# MAIN DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMP_DIR = BASE_DIR / "temp"


# ============================================================
# DATASETS
# ============================================================

ISIC2018_DIR = DATA_DIR / "ISIC2018"

ISIC_IMAGES = ISIC2018_DIR / "images"
ISIC_MASKS = ISIC2018_DIR / "masks"
ISIC_METADATA = ISIC2018_DIR / "metadata"


HAM10000_DIR = DATA_DIR / "HAM10000"

HAM_IMAGES = HAM10000_DIR / "images"
HAM_METADATA = HAM10000_DIR / "metadata"


# ============================================================
# MODEL DIRECTORIES
# ============================================================

CLASSIFICATION_MODEL_DIR = MODELS_DIR / "classification"
SEGMENTATION_MODEL_DIR = MODELS_DIR / "segmentation"
IMAGE_TYPE_MODEL_DIR = MODELS_DIR / "image_type"

ULTRALIGHT_VMUNET_DIR = Path(os.getenv("ULTRALIGHT_VMUNET_DIR", str(SEGMENTATION_MODEL_DIR / "ultralight_vmunet")))
CLIP_MODEL_DIR = Path(os.getenv("CLIP_MODEL_DIR", str(MODELS_DIR / "clip")))

# ============================================================
# EXTERNAL MODEL + API CONFIGURATION
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for directory in [
    DATA_DIR,
    MODELS_DIR,
    TEMP_DIR,
    ISIC_IMAGES,
    ISIC_MASKS,
    ISIC_METADATA,
    HAM_IMAGES,
    HAM_METADATA,
    CLASSIFICATION_MODEL_DIR,
    SEGMENTATION_MODEL_DIR,
    IMAGE_TYPE_MODEL_DIR,
    ULTRALIGHT_VMUNET_DIR,
    CLIP_MODEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


MODEL_RUNTIME_STATUS = {
    "ultralight_vmunet": {
        "env_var": "ULTRALIGHT_VMUNET_DIR",
        "path": str(ULTRALIGHT_VMUNET_DIR),
        "configured": bool(os.getenv("ULTRALIGHT_VMUNET_DIR")),
    },
    "clip": {
        "env_var": "CLIP_MODEL_DIR",
        "path": str(CLIP_MODEL_DIR),
        "configured": bool(os.getenv("CLIP_MODEL_DIR")),
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "api_url": DEEPSEEK_API_URL,
        "model": DEEPSEEK_MODEL,
        "configured": bool(DEEPSEEK_API_KEY and DEEPSEEK_API_URL),
    },
}
