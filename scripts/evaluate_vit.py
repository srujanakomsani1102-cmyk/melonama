from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import ViTForImageClassification, ViTImageProcessor


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "test.csv"
MODEL_DIR = BASE_DIR / "models" / "classification" / "vit_model"

IMAGE_DIRS = [
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_1",
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_2",
]

OUTPUT_DIR = BASE_DIR / "models" / "classification" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image(image_id: str) -> Path:
    for directory in IMAGE_DIRS:
        path = directory / f"{image_id}.jpg"
        if path.exists():
            return path

    raise FileNotFoundError(f"Image not found: {image_id}")


def main():
    print("=" * 70)
    print("CEFM ViT — Full HAM10000 Test Evaluation")
    print("=" * 70)

    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"ViT model not found: {MODEL_DIR}")

    # ------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------
    df = pd.read_csv(TEST_CSV)

    required_columns = {"image_id", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in test.csv: {sorted(missing)}")

    print(f"\nTest samples: {len(df)}")
    print(f"Non-melanoma: {(df['label'] == 0).sum()}")
    print(f"Melanoma:     {(df['label'] == 1).sum()}")

    # ------------------------------------------------------------
    # Load ViT
    # ------------------------------------------------------------
    print("\nLoading ViT model...")

    processor = ViTImageProcessor.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model = ViTForImageClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model: {MODEL_DIR}")

    # ------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------
    y_true = []
    y_pred = []
    y_prob = []

    print("\nRunning inference...")

    with torch.no_grad():

        for i, row in df.iterrows():

            image_path = find_image(str(row["image_id"]))

            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

            prediction = int(torch.argmax(probabilities).item())

            # Class 1 = melanoma
            melanoma_probability = float(probabilities[1].item())

            y_true.append(int(row["label"]))
            y_pred.append(prediction)
            y_prob.append(melanoma_probability)

            if (i + 1) % 100 == 0 or i + 1 == len(df):
                print(f"Processed {i + 1}/{len(df)}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        roc_auc = None

    # ------------------------------------------------------------
    # Confusion matrix
    #
    # [[TN, FP],
    #  [FN, TP]]
    # ------------------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    sensitivity = recall

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------
    results = {
        "model": "ViT-Melanoma",
        "backbone": "google/vit-base-patch16-224",
        "dataset": "HAM10000",
        "test_samples": int(len(y_true)),
        "non_melanoma_samples": int((y_true == 0).sum()),
        "melanoma_samples": int((y_true == 1).sum()),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"F1 Score:           {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:            {roc_auc:.4f}")
    else:
        print("ROC-AUC:            unavailable")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 NonMel   Mel")
    print(f"Actual NonMel    {tn:6d}  {fp:5d}")
    print(f"Actual Mel       {fn:6d}  {tp:5d}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["non_mel", "mel"],
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------
    results_path = OUTPUT_DIR / "vit_test_results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------
    prediction_df = df.copy()

    prediction_df["predicted_label"] = y_pred
    prediction_df["melanoma_probability"] = y_prob

    predictions_path = OUTPUT_DIR / "vit_test_predictions.csv"

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\nSaved:")
    print(results_path)
    print(predictions_path)


if __name__ == "__main__":
    main()