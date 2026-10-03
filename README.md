# CEFM Backend

Cross-modal Explainable Framework for Melanoma.

## Pipeline

1. Image validation
2. Image compatibility check
3. Lesion segmentation
4. ABCDE feature extraction (Asymmetry, Border, Color, **Diameter**)
5. Melanoma classification (trained model, or explainable ABCDE heuristic fallback)
6. **Evolution** comparison across visits of the same case
7. Explainability
8. Structured report

## Storage

Every uploaded image is saved under `data/uploads/` and its full analysis
(classification, ABCDE features incl. diameter, evolution, report) is stored
in the SQLite database `cefm.db` for future use.

## API (`/api/v1`)

| Method | Endpoint | Description |
|---|---|---|
| GET  | `/health` | Health check |
| POST | `/analyze` | Upload image + run full pipeline. Optional `case_id` / `case_label` form fields enable evolution tracking |
| POST | `/cases` | Create a case |
| GET  | `/cases` | List cases |
| GET  | `/cases/{case_id}` | Case detail with all analyses |
| GET  | `/analyses` | Recent analysis history |
| GET  | `/analyses/{id}` | Full stored result |
| GET  | `/images/{analysis_id}` | Retrieve a stored image |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
     -F "image=@lesion.jpg" -F "case_label=Left arm mole"
```

## Training

Train the classifier on HAM10000 (auto-locates the dataset):

```bash
python scripts/train_classifier.py --epochs 15 --batch-size 32
```

The best checkpoint is saved to `models/classification/best_model.pth`
and loaded automatically by the API on startup. Until a checkpoint exists,
the service falls back to an explainable heuristic based on ABCDE features
so any uploaded image always receives a result.

## Important

Do not use classifier outputs as medical diagnoses.

## Patient image quality and D measurement

Patient-uploaded images require a standardized **20 mm x 20 mm calibration marker** placed beside the lesion on the same skin surface. Images are rejected when required quality/calibration checks fail. The patient-reported approximate size is not used to calculate Diameter (D).

D is calculated from the segmented lesion's maximum pixel diameter and the marker-derived pixels-per-millimetre scale. No arbitrary pixel-to-mm fallback is used.

### Backend environment

Install `requirements.txt` in the backend virtual environment and run:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Streamlit environment

Use a separate virtual environment for Streamlit because the frontend and the pinned FastAPI backend have incompatible Starlette requirements.

```bash
python -m venv venv_frontend
venv_frontend\\Scripts\\activate
pip install -r requirements-streamlit.txt
python -m streamlit run streamlit_app.py
```

### Tests

```bash
python -m pytest -q
```
