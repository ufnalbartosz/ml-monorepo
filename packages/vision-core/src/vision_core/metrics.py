"""Accuracy over a classifier's output.

Both packages measured accuracy, in two different shapes: one compared
probabilities against one-hot labels, the other predicted classes in batches
and compared integers.  Both are here, as one batched predictor plus two thin
scoring functions, so the two packages agree on what a number means.
"""

from __future__ import annotations

import keras
import numpy as np

DEFAULT_BATCH_SIZE = 256


def predict_classes(
    model: keras.Model,
    images: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Predicted class-number per image, computed in batches to bound RAM use."""
    cls_pred = np.zeros(shape=len(images), dtype=np.int64)

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        predictions = model.predict(images[start:end], verbose=0)
        cls_pred[start:end] = np.argmax(predictions, axis=1)

    return cls_pred


def correct_predictions(cls_pred: np.ndarray, cls_true: np.ndarray) -> np.ndarray:
    """Boolean array of whether each prediction matched."""
    cls_pred = np.asarray(cls_pred)
    cls_true = np.asarray(cls_true)

    if len(cls_pred) != len(cls_true):
        raise ValueError(
            f"predictions and labels disagree on length: {len(cls_pred)} vs {len(cls_true)}"
        )

    return cls_true == cls_pred


def classification_accuracy(correct: np.ndarray) -> tuple[float, int]:
    """``(accuracy, number correct)`` from a boolean array.

    Averaging a boolean array counts False as 0 and True as 1, so the mean is
    the classification accuracy.
    """
    correct = np.asarray(correct)

    if correct.size == 0:
        raise ValueError("cannot compute accuracy over an empty array")

    return float(correct.mean()), int(correct.sum())


def accuracy_from_probabilities(
    predictions: np.ndarray,
    one_hot_labels: np.ndarray,
) -> float:
    """Fraction of rows whose argmax agrees, for probabilities vs one-hot labels."""
    predictions = np.asarray(predictions)
    one_hot_labels = np.asarray(one_hot_labels)

    if len(predictions) == 0:
        raise ValueError("cannot compute accuracy over zero predictions")
    if len(predictions) != len(one_hot_labels):
        raise ValueError(
            f"predictions and labels disagree on length: "
            f"{len(predictions)} vs {len(one_hot_labels)}"
        )

    return float(np.mean(np.argmax(predictions, axis=1) == np.argmax(one_hot_labels, axis=1)))
