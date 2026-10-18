"""The XGBoost fusion model.

Gradient-boosted trees are the right tool here and it is worth being explicit
about why, because "we used a neural network" is the reflex:

* **Native missing-value handling.** NFC is absent for most sessions, rPPG for
  many. XGBoost learns a default branch direction per split, so "no chip was
  read" is a fact the model can use rather than a hole to impute.
* **Mixed types.** Vendor scores, booleans, counts and a categorical platform,
  on wildly different scales, with no normalisation needed.
* **Monotone constraints.** A model that provably cannot decrease its risk
  estimate as the tamper score rises is one you can defend in an audit and one
  an attacker cannot walk downhill.
* **It is the honest baseline.** On tabular fraud data, boosted trees remain
  extremely hard to beat, and a deep model that appears to beat them is worth
  double-checking for leakage before it is worth deploying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from idv_risk.config import ModelConfig
from idv_risk.schema import FEATURE_NAMES, monotone_constraints


def compute_scale_pos_weight(labels: pd.Series | np.ndarray) -> float:
    """Ratio of negatives to positives, XGBoost's imbalance knob.

    Weighting positives by this ratio makes their contribution to the loss
    comparable to the negatives'. Returns 1.0 when there are no positives,
    rather than dividing by zero — a degenerate training set should fail on a
    clear error later, not here.
    """
    labels = np.asarray(labels)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    if positives == 0:
        return 1.0

    return float(negatives / positives)


def build_classifier(
    config: ModelConfig,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    scale_pos_weight: float | None = None,
) -> xgb.XGBClassifier:
    """Construct an unfitted classifier from the config and the schema.

    Note the constraints come from :mod:`idv_risk.schema`, not from this
    module: the direction a signal pushes risk is a property of the signal, and
    keeping it next to the signal's definition is what stops the two drifting
    apart.
    """
    params: dict = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "reg_lambda": config.reg_lambda,
        "reg_alpha": config.reg_alpha,
        "gamma": config.gamma,
        "eval_metric": config.eval_metric,
        "early_stopping_rounds": config.early_stopping_rounds,
        "tree_method": config.tree_method,
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "enable_categorical": True,
        "objective": "binary:logistic",
    }

    if config.scale_pos_weight is not None:
        params["scale_pos_weight"] = config.scale_pos_weight
    elif scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    if config.use_monotone_constraints:
        params["monotone_constraints"] = monotone_constraints(feature_names)

    return xgb.XGBClassifier(**params)


def fit_classifier(
    config: ModelConfig,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    valid_features: pd.DataFrame,
    valid_labels: pd.Series,
) -> xgb.XGBClassifier:
    """Fit with early stopping against the validation split.

    The validation split does double duty — early stopping here, and fitting
    the calibrator in :mod:`idv_risk.calibration`. That is a deliberate choice
    with a real cost: the calibrator sees data the model already stopped on, so
    its estimate is slightly optimistic. The alternative, a fourth split, costs
    positives that a fraud table cannot spare. The test split stays untouched
    by both, which is the property that actually matters.
    """
    feature_names = tuple(train_features.columns)

    classifier = build_classifier(
        config,
        feature_names=feature_names,
        scale_pos_weight=compute_scale_pos_weight(train_labels),
    )
    classifier.fit(
        train_features,
        train_labels,
        eval_set=[(valid_features, valid_labels)],
        verbose=False,
    )

    return classifier


def predict_risk(classifier: xgb.XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    """P(fraud) for each row.

    When the model was fitted with early stopping, XGBoost's ``predict_proba``
    already truncates to ``best_iteration``; the trees after it are still in the
    booster but are not evaluated. Verified by ``test_model.py`` rather than
    assumed, because it is version-dependent behaviour that silently changes
    the answer if it ever stops holding.
    """
    return classifier.predict_proba(features)[:, 1]


def feature_importance(
    classifier: xgb.XGBClassifier,
    importance_type: str = "gain",
) -> pd.Series:
    """Per-feature importance, descending.

    ``gain`` rather than the ``weight`` default: weight counts how often a
    feature was split on, which flatters high-cardinality continuous signals
    and understates a decisive boolean like ``virtual_camera_detected``.
    """
    booster = classifier.get_booster()
    scores = booster.get_score(importance_type=importance_type)

    names = list(getattr(classifier, "feature_names_in_", [])) or sorted(scores)
    series = pd.Series({name: scores.get(name, 0.0) for name in names}, dtype="float64")

    return series.sort_values(ascending=False)
