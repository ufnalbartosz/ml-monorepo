"""Batched prediction and accuracy reporting.

These functions used to sit in ``main.py`` and read the module-level ``x``,
``y_true``, ``y_pred_cls`` and ``session`` globals, so they could only be
called from inside that one script.  They now take a ``keras.Model`` and the
arrays they operate on.
"""

from __future__ import annotations

import keras
import numpy as np

from project_cnn import plot

DEFAULT_BATCH_SIZE = 256


def predict_cls(
    model: keras.Model,
    images: np.ndarray,
    cls_true: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict classes in batches to limit RAM usage.

    Returns ``(correct, cls_pred)``: a boolean array of whether each image was
    classified correctly, and the predicted class-numbers.
    """
    if len(images) != len(cls_true):
        raise ValueError(f"images and labels disagree on length: {len(images)} vs {len(cls_true)}")

    cls_pred = np.zeros(shape=len(images), dtype=np.int64)

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        predictions = model.predict(images[start:end], verbose=0)
        cls_pred[start:end] = np.argmax(predictions, axis=1)

    return cls_true == cls_pred, cls_pred


def classification_accuracy(correct: np.ndarray) -> tuple[float, int]:
    """Accuracy and the number of correct classifications.

    Averaging a boolean array counts False as 0 and True as 1, so the mean is
    the classification accuracy.
    """
    correct = np.asarray(correct)

    if correct.size == 0:
        raise ValueError("cannot compute accuracy over an empty array")

    return float(correct.mean()), int(correct.sum())


def evaluate_split(
    model: keras.Model,
    dataset: dict,
    split: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Accuracy of ``model`` on one of the 'train'/'valid'/'test' splits."""
    correct, cls_pred = predict_cls(
        model,
        images=dataset[f"{split}_images"],
        cls_true=dataset[f"{split}_cls"],
        batch_size=batch_size,
    )
    accuracy, _ = classification_accuracy(correct)

    return accuracy, correct, cls_pred


def print_valid_accuracy(
    model: keras.Model,
    dataset: dict,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    accuracy, correct, _ = evaluate_split(model, dataset, "valid", batch_size)

    print(f"Accuracy on Validation-Set: {accuracy:.1%} ({int(correct.sum())} / {len(correct)})")

    return accuracy


def print_test_accuracy(
    model: keras.Model,
    dataset: dict,
    show_example_errors: bool = False,
    show_confusion_matrix: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> float:
    accuracy, correct, cls_pred = evaluate_split(model, dataset, "test", batch_size)

    print(f"Accuracy on Test-Set: {accuracy:.1%} ({int(correct.sum())} / {len(correct)})")

    if show_example_errors:
        print("Example errors:")
        plot.plot_example_errors(cls_pred=cls_pred, correct=correct, dataset=dataset)

    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot.plot_confusion_matrix(cls_pred=cls_pred, dataset=dataset)

    return accuracy
