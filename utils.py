"""
Shared utility functions and configuration for CNN pill classification notebooks.
Includes image I/O, metadata loading, data splitting, visualization, and evaluation.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ────────────────────────────────────────────────────────────────
#   SHARED CONFIGURATION
# ────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Unified configuration for both Siamese and ResNet notebooks."""

    # Required inputs
    metadata_csv: str = "image_meta.csv"
    file_root: str = "."  # base directory for metadata, images, and outputs
    image_path_col: str = "image_path"  # column name in CSV for image paths
    label_col: str = "label"  # column name in CSV for class labels

    # Dataset controls
    image_size: int = 224  # all images resized to this square size
    channels: int = 3  # 1 for grayscale (Siamese), 3 for RGB (ResNet)
    test_size: float = 0.20  # fraction of data for testing
    val_size: float = 0.15  # fraction for validation (used in supervised learning)
    min_images_per_class: int = 5  # minimum images per class to include
    max_classes: int | None = None  # limit classes (None = use all eligible)

    # Training parameters
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 0.001
    seed: int = 42

    # Siamese-specific parameters
    train_steps_per_epoch: int = 200
    val_pairs_per_class: int = 30
    embedding_dim: int = 64
    threshold_grid_size: int = 101

    # Image Augmentation
    random_flip: str = "horizontal_and_vertical"
    random_rotation: float = 0.08
    random_zoom: float = 0.10

    # Output
    output_subdir: str = "outputs"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.utils import to_categorical
import tensorflow as tf


# ────────────────────────────────────────────────────────────────
#   REPRODUCIBILITY
# ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ────────────────────────────────────────────────────────────────
#   IMAGE I/O
# ────────────────────────────────────────────────────────────────

def load_and_preprocess_image(
    path: str,
    image_size: int,
    channels: int = 3,
) -> np.ndarray:
    """
    Load and preprocess a single image.

    Args:
        path: Path to the image file.
        image_size: Target square size (image_size x image_size).
        channels: Number of channels (1 for grayscale, 3 for RGB).

    Returns:
        Preprocessed image as float32 array, normalized to [0, 1].

    Raises:
        ValueError: If image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # Convert to requested channels
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")

    # Resize to target size
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Add channel dimension if grayscale
    if channels == 1:
        img = np.expand_dims(img, axis=-1)

    # Normalize to [0, 1]
    img = img.astype("float32") / 255.0
    return img


def load_image_batch(
    paths: List[str],
    image_size: int,
    channels: int = 3,
) -> np.ndarray:
    """
    Load and preprocess a batch of images.

    Args:
        paths: List of image file paths.
        image_size: Target square size.
        channels: Number of channels (1 or 3).

    Returns:
        Stack of preprocessed images as float32 array.
    """
    images = [
        load_and_preprocess_image(path, image_size, channels)
        for path in paths
    ]
    return np.stack(images).astype("float32")


# ────────────────────────────────────────────────────────────────
#   METADATA & DATA LOADING
# ────────────────────────────────────────────────────────────────

def load_metadata(
    metadata_csv: str,
    image_col: str,
    label_col: str,
    file_root: str,
    min_images_per_class: int = 5,
    max_classes: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and filter metadata CSV.

    Args:
        metadata_csv: Path to CSV file or filename (relative to file_root).
        image_col: Name of column containing image paths.
        label_col: Name of column containing class labels.
        file_root: Base directory for resolving relative image paths.
        min_images_per_class: Minimum images per class to be included.
        max_classes: If set, randomly sample this many classes.
        seed: Random seed for class sampling.

    Returns:
        Filtered DataFrame with columns: image_col, label_col, full_path.

    Raises:
        FileNotFoundError: If metadata CSV not found.
        ValueError: If required columns missing or no usable rows remain.
    """
    # Resolve CSV path
    if not Path(metadata_csv).is_absolute():
        metadata_csv = str(Path(file_root) / metadata_csv)

    metadata_path = Path(metadata_csv)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata CSV: {metadata_path.resolve()}\n"
            f"Update metadata_csv or file_root."
        )

    df = pd.read_csv(metadata_path).copy()

    # Validate required columns
    required = {image_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Keep only rows with non-null values
    df = df.dropna(subset=[image_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str)

    # Build full paths
    image_root = Path(file_root)
    df["full_path"] = df[image_col].apply(
        lambda p: str((image_root / str(p)).resolve())
    )

    # Filter to images that exist
    df = df[df["full_path"].apply(lambda p: Path(p).exists())].copy()

    # Filter classes with enough images
    counts = df[label_col].value_counts()
    eligible_labels = counts[counts >= min_images_per_class].index.tolist()
    df = df[df[label_col].isin(eligible_labels)].copy()

    # Optionally sample max_classes
    if max_classes is not None:
        unique_labels = df[label_col].drop_duplicates().tolist()
        rng = np.random.default_rng(seed)
        sampled = rng.choice(
            unique_labels,
            size=min(max_classes, len(unique_labels)),
            replace=False,
        ).tolist()
        df = df[df[label_col].isin(sampled)].copy()

    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(
            "No usable rows remain after filtering. "
            "Check paths and min_images_per_class."
        )

    return df


def encode_labels(
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> Tuple[Dict[str, int], Dict[int, str], np.ndarray, np.ndarray]:
    """
    Create label encoder and encode train/test labels consistently.

    Args:
        train_labels: Label column from training split.
        test_labels: Label column from test split.

    Returns:
        Tuple of:
            - label_to_id: Dict mapping string labels to integer IDs.
            - id_to_label: Dict mapping integer IDs back to string labels.
            - train_y: Encoded training labels.
            - test_y: Encoded test labels.
    """
    unique_labels = sorted(
        pd.concat([train_labels, test_labels]).astype(str).unique().tolist()
    )
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_y = train_labels.astype(str).map(label_to_id).to_numpy(dtype=np.int32)
    test_y = test_labels.astype(str).map(label_to_id).to_numpy(dtype=np.int32)

    return label_to_id, id_to_label, train_y, test_y


def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train/val/test with stratification.

    Args:
        df: Input DataFrame.
        label_col: Column name for stratification.
        test_size: Fraction for test set (of total).
        val_size: Fraction for validation set (of total).
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # First: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        random_state=seed,
        stratify=df[label_col],
    )

    # Second: val vs test (relative to temp)
    relative_test_size = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df[label_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_dataset_arrays(
    split_df: pd.DataFrame,
    image_size: int,
    channels: int = 3,
    label_encoder: LabelEncoder | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images and encode labels for a split.

    Args:
        split_df: DataFrame with 'full_path' and label columns.
        image_size: Target image size.
        channels: Number of image channels (1 or 3).
        label_encoder: Pre-fitted LabelEncoder. If None, assumes numeric labels.

    Returns:
        Tuple of (X, y_int, Y_onehot):
            - X: Stacked images array.
            - y_int: Integer-encoded labels.
            - Y_onehot: One-hot encoded labels.
    """
    images = []
    labels = []

    for _, row in split_df.iterrows():
        try:
            img = load_and_preprocess_image(
                row["full_path"],
                image_size,
                channels,
            )
            images.append(img)
            # Try to get label from multiple possible column names
            label = row.get("label") or row.get(row.columns[-1])
            labels.append(label)
        except Exception as e:
            print(f"Skipping {row.get('full_path', 'unknown')}: {e}")

    X = np.stack(images)

    if label_encoder is not None:
        y_int = label_encoder.transform(labels)
        num_classes = len(label_encoder.classes_)
    else:
        y_int = np.array(labels, dtype=np.int32)
        num_classes = int(np.max(y_int)) + 1

    Y_onehot = to_categorical(y_int, num_classes=num_classes)

    return X, y_int, Y_onehot


# ────────────────────────────────────────────────────────────────
#   VISUALIZATION
# ────────────────────────────────────────────────────────────────

def show_image_examples(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    n: int = 8,
    cmap: str | None = None,
) -> None:
    """
    Display random image examples with their class labels.

    Args:
        X: Batch of images (N, H, W, C).
        y: Integer class labels.
        class_names: List of class name strings.
        n: Number of examples to show.
        cmap: Colormap for grayscale images (e.g., 'gray'). Auto-detect for RGB.
    """
    idxs = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    cols = 4
    rows = math.ceil(len(idxs) / cols)

    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, idx in enumerate(idxs, start=1):
        plt.subplot(rows, cols, i)
        img = X[idx].squeeze()
        # Auto-detect: use cmap only for 2D grayscale images
        imshow_cmap = cmap if img.ndim == 2 else None
        plt.imshow(img, cmap=imshow_cmap)
        plt.title(class_names[y[idx]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_prediction_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    n: int = 8,
    correct: bool = False,
    cmap: str | None = None,
) -> None:
    """
    Display prediction examples (correct or incorrect).

    Args:
        X: Batch of images.
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        y_prob: Predicted probabilities (N, num_classes).
        class_names: List of class names.
        n: Number of examples to show.
        correct: If True, show correct predictions; else incorrect.
        cmap: Colormap for grayscale images. Auto-detect for RGB.
    """
    matches = np.where(
        (y_true == y_pred) if correct else (y_true != y_pred)
    )[0]

    if len(matches) == 0:
        print(
            f"No {'correct' if correct else 'incorrect'} "
            f"predictions to display."
        )
        return

    chosen = np.random.choice(
        matches,
        size=min(n, len(matches)),
        replace=False,
    )
    cols = 4
    rows = math.ceil(len(chosen) / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, idx in enumerate(chosen, start=1):
        plt.subplot(rows, cols, i)
        img = X[idx].squeeze()
        imshow_cmap = cmap if img.ndim == 2 else None
        plt.imshow(img, cmap=imshow_cmap)

        pred = y_pred[idx]
        true = y_true[idx]
        conf = y_prob[idx, pred]
        color = "green" if pred == true else "red"

        plt.title(
            f"True: {class_names[true]}\n"
            f"Pred: {class_names[pred]} ({conf:.2%})",
            color=color,
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────
#   TRAINING & HISTORY
# ────────────────────────────────────────────────────────────────

def plot_training_history(
    history_dict: Dict[str, List[float]],
    metrics_to_plot: List[Tuple[str, str]] | None = None,
) -> None:
    """
    Plot training history with train/val curves.

    Args:
        history_dict: Dictionary with keys like 'loss', 'val_loss', etc.
        metrics_to_plot: List of (train_key, val_key) tuples to plot.
                        If None, uses defaults: loss, accuracy, auc metrics.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            ("loss", "val_loss"),
            ("accuracy", "val_accuracy"),
        ]
        # Add auc metrics if present
        if "roc_auc" in history_dict:
            metrics_to_plot.append(("roc_auc", "val_roc_auc"))
        if "pr_auc" in history_dict:
            metrics_to_plot.append(("pr_auc", "val_pr_auc"))
        if "top5_accuracy" in history_dict:
            metrics_to_plot.append(("top5_accuracy", "val_top5_accuracy"))

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))

    if n == 1:
        axes = [axes]

    for ax, (train_key, val_key) in zip(axes, metrics_to_plot):
        if train_key in history_dict:
            ax.plot(history_dict[train_key], label=train_key)
        if val_key in history_dict:
            ax.plot(history_dict[val_key], label=val_key)

        title = train_key.replace("_", " ").title()
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────────
#   EVALUATION METRICS
# ────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    grid_size: int = 101,
) -> Tuple[float, pd.DataFrame]:
    """
    Find optimal classification threshold by grid search.

    Args:
        y_true: Binary true labels {0, 1}.
        y_prob: Predicted probabilities [0, 1].
        metric: Metric to optimize: 'f1', 'accuracy', 'precision', 'recall'.
        grid_size: Number of thresholds to evaluate.

    Returns:
        Tuple of (best_threshold, results_table).
    """
    thresholds = np.linspace(0.0, 1.0, grid_size)
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    results = pd.DataFrame(rows).sort_values(
        [metric, "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    best_threshold = float(results.loc[0, "threshold"])
    return best_threshold, results


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Binary true labels.
        y_pred: Binary predictions.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with: accuracy, precision, recall, f1, roc_auc, pr_auc.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "",
) -> None:
    """
    Plot ROC curve and Precision-Recall curve.

    Args:
        y_true: Binary true labels.
        y_prob: Predicted probabilities.
        title: Optional title for the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ROC Curve
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] | None = None,
) -> None:
    """
    Print classification report and confusion matrix.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        class_names: Optional list of class names.
    """
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    if class_names:
        indices = class_names
    else:
        indices = [f"Class {i}" for i in range(len(cm))]

    cm_df = pd.DataFrame(cm, index=indices, columns=indices)
    print("\nConfusion Matrix:")
    print(cm_df)
