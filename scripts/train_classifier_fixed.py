from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ViTForImageClassification, ViTImageProcessor


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "train.csv"
VAL_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "validation.csv"

IMAGE_DIRS = [
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_1",
    BASE_DIR / "datasets" / "Ham10000" / "HAM10000_images_part_2",
]

MODEL_BACKBONE = "google/vit-base-patch16-224"
IMAGE_SIZE = 224

# Separate experimental model directory.
OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "classification"
    / "vit_model_experiment"
)


def find_image(image_id: str) -> Path:
    """Find a HAM10000 image in either image directory."""

    for directory in IMAGE_DIRS:
        path = directory / f"{image_id}.jpg"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Image not found in HAM10000 image directories: {image_id}"
    )


def load_split(csv_path: Path) -> pd.DataFrame:
    """Load an existing prepared split."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {"image_id", "label"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing)}"
        )

    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)


class HAMDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        augment: bool = False,
    ):
        self.frame = frame
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        row = self.frame.iloc[idx]

        image_path = find_image(
            str(row["image_id"])
        )

        image = Image.open(image_path).convert("RGB")

        if self.augment:

            if np.random.rand() < 0.5:
                image = ImageOps.mirror(image)

            if np.random.rand() < 0.5:
                image = image.rotate(
                    np.random.uniform(-20, 20),
                    resample=Image.BILINEAR,
                )

            if np.random.rand() < 0.25:
                image = ImageOps.autocontrast(image)

        label = int(row["label"])

        return image, label


def make_balanced_sampler(labels):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels,
        minlength=2,
    ).astype(np.float32)

    weights = (
        np.max(counts)
        / np.clip(counts, 1.0, None)
    )

    sample_weights = weights[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_model():

    model = ViTForImageClassification.from_pretrained(
        MODEL_BACKBONE
    )

    model.config.num_labels = 2

    model.config.id2label = {
        0: "non_mel",
        1: "mel",
    }

    model.config.label2id = {
        "non_mel": 0,
        "mel": 1,
    }

    model.num_labels = 2

    model.classifier = torch.nn.Linear(
        model.config.hidden_size,
        2,
    )

    model.config.problem_type = (
        "single_label_classification"
    )

    return model


def train(args):

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 70)
    print("CEFM — ViT Fixed-Split Training")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load the EXISTING prepared splits
    # ---------------------------------------------------------

    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)

    print("\nDataset:")
    print(f"Train images:      {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    print("\nTrain distribution:")
    print(train_df["label"].value_counts().sort_index())

    print("\nValidation distribution:")
    print(val_df["label"].value_counts().sort_index())

    # ---------------------------------------------------------
    # Verify all images before expensive training
    # ---------------------------------------------------------

    print("\nChecking image paths...")

    for frame_name, frame in [
        ("train", train_df),
        ("validation", val_df),
    ]:

        missing = 0

        for image_id in frame["image_id"]:

            try:
                find_image(str(image_id))

            except FileNotFoundError:
                missing += 1

        print(
            f"{frame_name}: "
            f"{len(frame) - missing}/{len(frame)} images found"
        )

        if missing:
            raise RuntimeError(
                f"{missing} images are missing from {frame_name}."
            )

    # ---------------------------------------------------------
    # Processor
    # ---------------------------------------------------------

    print("\nLoading ViT processor...")

    processor = ViTImageProcessor.from_pretrained(
        MODEL_BACKBONE
    )

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------

    train_dataset = HAMDataset(
        train_df,
        augment=True,
    )

    val_dataset = HAMDataset(
        val_df,
        augment=False,
    )

    sampler = make_balanced_sampler(
        train_df["label"].to_numpy()
    )

    def collate_fn(batch):

        images = [
            item[0]
            for item in batch
        ]

        labels = torch.tensor(
            [
                item[1]
                for item in batch
            ],
            dtype=torch.long,
        )

        encoded = processor(
            images=images,
            return_tensors="pt",
        )

        encoded["labels"] = labels

        return encoded

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nLoading pretrained ViT...")

    model = build_model()

    # Freeze backbone for the first controlled experiment.
    if args.freeze_backbone:

        print(
            "Freezing ViT backbone; "
            "training classifier head only."
        )

        for param in model.vit.parameters():
            param.requires_grad = False

        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True

    else:

        print(
            "Full ViT fine-tuning enabled."
        )

        for param in model.parameters():
            param.requires_grad = True

    model.to(device)

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in trainable_params):,}",
    )

    # ---------------------------------------------------------
    # Loss / optimizer
    # ---------------------------------------------------------

    counts = np.bincount(
        train_df["label"].to_numpy(),
        minlength=2,
    ).astype(np.float32)

    class_weights = (
        np.max(counts)
        / np.clip(counts, 1.0, None)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(args.epochs):

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        # -----------------------------
        # Training
        # -----------------------------

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_seen = 0

        for batch_index, batch in enumerate(
            train_loader
        ):

            if (
                args.max_steps > 0
                and batch_index >= args.max_steps
            ):
                break

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = criterion(
                outputs.logits,
                batch["labels"],
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * len(batch["labels"])
            )

            predictions = (
                outputs.logits.argmax(dim=1)
            )

            train_correct += (
                predictions
                == batch["labels"]
            ).sum().item()

            train_seen += len(
                batch["labels"]
            )

            if (
                batch_index + 1
            ) % 100 == 0:

                print(
                    f"  Train batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)}"
                )

        train_loss /= max(
            train_seen,
            1,
        )

        train_accuracy = (
            train_correct
            / max(train_seen, 1)
        )

        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_seen = 0

        with torch.no_grad():

            for batch in val_loader:

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                outputs = model(**batch)

                loss = criterion(
                    outputs.logits,
                    batch["labels"],
                )

                val_loss += (
                    loss.item()
                    * len(batch["labels"])
                )

                predictions = (
                    outputs.logits.argmax(dim=1)
                )

                val_correct += (
                    predictions
                    == batch["labels"]
                ).sum().item()

                val_seen += len(
                    batch["labels"]
                )

        val_loss /= max(
            val_seen,
            1,
        )

        val_accuracy = (
            val_correct
            / max(val_seen, 1)
        )

        print(
            f"\nTrain loss: {train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy:.4f}"
        )

        # -----------------------------------------------------
        # Save best experimental model
        # -----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            model.save_pretrained(
                OUTPUT_DIR
            )

            processor.save_pretrained(
                OUTPUT_DIR
            )

            print(
                f"\nSaved best experimental model:"
            )

            print(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Experimental model: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help=(
            "Optional batch limit. "
            "0 means full epoch."
        ),
    )

    train(
        parser.parse_args()
    )