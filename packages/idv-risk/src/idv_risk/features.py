"""The one function that turns a raw application into model input.

**Training and serving must call this same function.** A resize interpolation
that differs between train and serve is the classic vision bug; the tabular
equivalent is a column order that differs, a category encoded two ways, or a
missing value imputed in training and left NaN in production. All three are
silent — the model returns a number either way, just the wrong one.

So this module is deliberately small and has no configuration: given a frame of
raw signals, there is exactly one output. Everything it needs to know lives in
:mod:`idv_risk.schema`.

Three things it guarantees, each of which is a bug that would otherwise reach
production:

1. **Column order** matches :data:`~idv_risk.schema.FEATURE_NAMES`. XGBoost
   matches by position when handed a bare array.
2. **Unknown categorical levels are folded** into ``__unknown__``. XGBoost
   raises on an unseen level, so a new device platform would take scoring down
   rather than degrade it.
3. **Missing optional signals stay NaN.** They are not imputed: XGBoost learns
   a default direction per split, and "no NFC chip was read" is information,
   not a gap to paper over with a mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idv_risk.schema import (
    FEATURE_NAMES,
    SIGNALS_BY_NAME,
    UNKNOWN_CATEGORY,
    Signal,
)


class MissingSignalError(KeyError):
    """A required signal was absent from the input frame."""


def build_features(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> pd.DataFrame:
    """Project ``frame`` onto the model's input contract.

    :param frame: raw signals, one row per application. Extra columns are
        ignored, so the warehouse table can be passed straight in.
    :param feature_names: subset of the contract to build. Only the ablation
        experiments pass anything but the default.
    :raises MissingSignalError: if a non-optional signal is absent. Optional
        signals may be missing entirely and become an all-NaN column, which is
        what happens when a whole capability (say NFC) is not deployed yet.
    """
    missing_required = [
        name
        for name in feature_names
        if name not in frame.columns and not SIGNALS_BY_NAME[name].optional
    ]
    if missing_required:
        raise MissingSignalError(f"required signals absent from input: {sorted(missing_required)}")

    columns = {name: _build_column(frame, SIGNALS_BY_NAME[name]) for name in feature_names}

    # dict ordering is insertion order, so this fixes the column order to
    # feature_names. That is load-bearing, not incidental.
    return pd.DataFrame(columns, index=frame.index)


def _build_column(frame: pd.DataFrame, signal: Signal) -> pd.Series:
    if signal.name not in frame.columns:
        # Only reachable for optional signals; the guard above rejects the rest.
        if signal.is_categorical:
            return pd.Series(
                pd.Categorical([UNKNOWN_CATEGORY] * len(frame), categories=signal.categories),
                index=frame.index,
            )
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    values = frame[signal.name]

    if signal.is_categorical:
        return _build_categorical(values, signal)

    return _build_numeric(values, signal)


def _build_categorical(values: pd.Series, signal: Signal) -> pd.Series:
    """Coerce to the schema's closed level set, folding anything unseen.

    Without this, a device platform that did not exist at training time makes
    ``predict`` raise. Folding to a known level degrades the prediction for
    that row instead of failing the request.
    """
    categories = signal.categories
    assert categories is not None  # guarded by caller

    text = values.astype("string")
    known = text.isin(list(categories))
    folded = text.where(known, UNKNOWN_CATEGORY)
    # A genuinely absent value stays absent rather than becoming "unknown":
    # "we did not collect this" and "we collected something we do not
    # recognise" are different facts, and trees can split on the difference.
    folded = folded.where(~text.isna(), None)

    return pd.Series(
        pd.Categorical(folded, categories=categories),
        index=values.index,
    )


def _build_numeric(values: pd.Series, signal: Signal) -> pd.Series:
    """Coerce to float and clip to the declared range.

    Clipping rather than rejecting is deliberate: an upstream vendor shipping a
    score of 1.02 should not take the request down, and a value outside the
    declared range is outside the training distribution anyway, so the clipped
    prediction is the honest one.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")

    if signal.minimum is not None or signal.maximum is not None:
        numeric = numeric.clip(lower=signal.minimum, upper=signal.maximum)

    return numeric


def validate_features(
    features: pd.DataFrame, feature_names: tuple[str, ...] = FEATURE_NAMES
) -> None:
    """Assert a built frame satisfies the contract.

    Cheap enough to call on every scoring request, and the thing that turns a
    silent wrong answer into a loud error.
    """
    actual = tuple(features.columns)
    if actual != feature_names:
        raise ValueError(
            f"feature order mismatch:\n  expected {feature_names}\n  got      {actual}"
        )

    for name in feature_names:
        signal = SIGNALS_BY_NAME[name]
        column = features[name]

        if signal.is_categorical:
            if not isinstance(column.dtype, pd.CategoricalDtype):
                raise ValueError(f"{name}: expected categorical dtype, got {column.dtype}")
            unexpected = set(column.dtype.categories) - set(signal.categories or ())
            if unexpected:
                raise ValueError(f"{name}: unexpected categories {sorted(unexpected)}")
            continue

        if not pd.api.types.is_float_dtype(column):
            raise ValueError(f"{name}: expected float dtype, got {column.dtype}")

        if not signal.optional and column.isna().any():
            raise ValueError(
                f"{name}: required signal has {int(column.isna().sum())} missing values"
            )
