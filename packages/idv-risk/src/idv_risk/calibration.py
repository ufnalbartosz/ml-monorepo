"""Turning raw model scores into probabilities that mean what they say.

This is not a nicety here, it is a dependency. :mod:`idv_risk.decision` picks
its accept/step-up/reject boundaries by minimising expected cost, and that
arithmetic multiplies a probability by a cost. If the model says 0.30 and the
true rate at that score is 0.08, every threshold derived from it is wrong.

Boosted ensembles trained with ``scale_pos_weight`` are doubly miscalibrated:
gradient boosting pushes scores toward the extremes, and the positive
reweighting inflates them further. So the raw score is a good *ranking* and a
bad *probability* — which is exactly the situation calibration exists for.

The method choice follows the data volume. Isotonic regression is
non-parametric and fits any monotone distortion, but with few positives it
interpolates noise; sigmoid (Platt) has two parameters and cannot. The
threshold is in :class:`~idv_risk.config.CalibrationConfig`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from idv_risk.config import CalibrationConfig


def choose_method(
    labels: pd.Series | np.ndarray,
    config: CalibrationConfig,
) -> str:
    """Pick isotonic or sigmoid based on how much calibration data there is."""
    labels = np.asarray(labels)
    n_samples = len(labels)
    n_positives = int((labels == 1).sum())

    if n_samples < config.min_samples or n_positives < config.min_positives:
        return config.fallback_method

    return config.method


def fit_calibrator(
    classifier,
    features: pd.DataFrame,
    labels: pd.Series,
    config: CalibrationConfig | None = None,
) -> CalibratedClassifierCV:
    """Wrap a fitted classifier in a calibration layer.

    ``FrozenEstimator`` is what stops ``CalibratedClassifierCV`` from refitting
    the underlying model — it replaced the old ``cv="prefit"`` argument, which
    scikit-learn deprecated in 1.6. Without it the classifier would be cloned
    and retrained on the calibration split, discarding the early-stopped fit
    and silently training on data meant for calibration.

    The caller is responsible for ``features`` being disjoint from what the
    classifier was fitted on; :func:`idv_risk.pipeline.train` uses the
    validation split.
    """
    config = config or CalibrationConfig()
    method = choose_method(labels, config)

    calibrator = CalibratedClassifierCV(FrozenEstimator(classifier), method=method)
    calibrator.fit(features, labels)

    return calibrator


def predict_calibrated(calibrator: CalibratedClassifierCV, features: pd.DataFrame) -> np.ndarray:
    """Calibrated P(fraud) per row."""
    return calibrator.predict_proba(features)[:, 1]


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error: mean |confidence - accuracy| across bins.

    Bins are equal-width in probability, weighted by occupancy. Reported
    alongside Brier score, because ECE alone can be gamed by a model that
    predicts the base rate for everything — such a model is perfectly
    calibrated and completely useless, which is why discrimination
    (PR-AUC) and calibration always get reported together.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    if len(probabilities) == 0:
        raise ValueError("cannot compute calibration error over zero predictions")
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels disagree on length: {len(probabilities)} vs {len(labels)}"
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts values equal to an edge in the upper bin; clip so that
    # p == 1.0 lands in the last bin rather than one past it.
    bin_index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)

    error = 0.0
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(probabilities[in_bin].mean())
        observed = float(labels[in_bin].mean())
        error += (count / len(probabilities)) * abs(confidence - observed)

    return error


def reliability_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin predicted vs observed fraud rate.

    The thing to actually look at. A single ECE number hides which end of the
    range is wrong, and for a risk engine the high-score end is the one whose
    calibration decides whether the reject threshold is where you think.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        count = int(in_bin.sum())
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "count": count,
                "mean_predicted": float(probabilities[in_bin].mean()) if count else np.nan,
                "observed_rate": float(labels[in_bin].mean()) if count else np.nan,
            }
        )

    return pd.DataFrame(rows)
