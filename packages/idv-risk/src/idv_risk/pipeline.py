"""The training pipeline, start to finish.

Split -> build features -> fit -> calibrate -> choose thresholds -> evaluate ->
bundle. Each step is a function taking its inputs explicitly, and this module
only wires them together, so any step can be exercised on its own.

The ordering constraint worth stating plainly: **the test split is not touched
until the final evaluation.** The model early-stops on validation, the
calibrator fits on validation, and the thresholds are optimised on validation.
If the thresholds were tuned on test, the reported cost saving would be the
best case rather than the expected one, and that is the number a business
decision gets made on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from idv_risk.artifact import Provenance, RiskModel
from idv_risk.calibration import fit_calibrator, reliability_table
from idv_risk.config import TrainingConfig
from idv_risk.decision import baseline_costs, decision_report, optimise_thresholds
from idv_risk.features import build_features
from idv_risk.metrics import evaluation_summary, recall_by_group
from idv_risk.model import feature_importance, fit_classifier
from idv_risk.schema import FEATURE_NAMES, LABEL, SLICE_COLUMNS
from idv_risk.splits import Split, assert_no_group_overlap, time_grouped_split


@dataclass
class TrainingResult:
    """The model plus everything learned about it on the way."""

    model: RiskModel
    split: Split
    validation_report: dict[str, Any]
    test_report: dict[str, Any]

    @property
    def test_pr_auc(self) -> float:
        return self.test_report["overall"]["pr_auc"]


def train(
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    max_step_up_rate: float | None = 0.20,
) -> TrainingResult:
    """Train a risk model on a labelled application table.

    :param frame: raw applications — signals, identifiers, slice columns and
        the label. Feature building happens inside, so the caller cannot use a
        different path from the one serving will use.
    :param max_step_up_rate: operational cap on the review band; see
        :func:`idv_risk.decision.optimise_thresholds`.
    """
    config = config or TrainingConfig()

    if LABEL not in frame.columns:
        raise KeyError(f"frame has no {LABEL!r} column")

    split = time_grouped_split(
        frame,
        group_columns=config.group_columns,
        time_column=config.time_column,
        valid_fraction=config.valid_fraction,
        test_fraction=config.test_fraction,
    )
    # Cheap, and the failure it catches — a ring straddling the boundary —
    # inflates every number downstream without looking like a bug.
    assert_no_group_overlap(frame, split, config.group_columns)

    train_frame, valid_frame, test_frame = split.frames(frame)

    train_x = build_features(train_frame, feature_names)
    valid_x = build_features(valid_frame, feature_names)
    test_x = build_features(test_frame, feature_names)

    train_y = train_frame[LABEL]
    valid_y = valid_frame[LABEL]
    test_y = test_frame[LABEL]

    classifier = fit_classifier(config.model, train_x, train_y, valid_x, valid_y)
    calibrator = fit_calibrator(classifier, valid_x, valid_y, config.calibration)

    valid_probabilities = calibrator.predict_proba(valid_x)[:, 1]
    thresholds = optimise_thresholds(
        valid_probabilities,
        valid_y.to_numpy(),
        config.cost,
        max_step_up_rate=max_step_up_rate,
    )

    test_probabilities = calibrator.predict_proba(test_x)[:, 1]

    # Metrics are reported at the *flag* threshold - the boundary above which
    # an application stops being auto-accepted. That is the decision a genuine
    # applicant feels and the one an attacker has to get under, so it is the
    # threshold BPCER and the subgroup rates should be quoted at. The reject
    # boundary sits far to the right and touches too few rows to estimate a
    # per-group error rate from.
    validation_report = _build_report(
        valid_probabilities, valid_frame, thresholds.accept_below, config
    )
    test_report = _build_report(test_probabilities, test_frame, thresholds.accept_below, config)
    test_report["feature_importance"] = feature_importance(classifier).head(15).to_dict()
    test_report["reliability"] = reliability_table(test_probabilities, test_y.to_numpy()).to_dict(
        orient="records"
    )

    model = RiskModel(
        classifier=classifier,
        calibrator=calibrator,
        thresholds=thresholds,
        feature_names=feature_names,
        config=config,
        provenance=Provenance.build(
            train_frame,
            LABEL,
            metrics={"test": test_report["overall"], "validation": validation_report["overall"]},
        ),
    )

    return TrainingResult(
        model=model,
        split=split,
        validation_report=validation_report,
        test_report=test_report,
    )


def _build_report(
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    threshold: float,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Metrics, decision bands and cost baselines for one split."""
    labels = frame[LABEL].to_numpy()

    slices = {
        name: frame[name]
        for name in SLICE_COLUMNS
        if name in frame.columns and name != "fraud_type"
    }

    report = evaluation_summary(probabilities, labels, threshold, slices=slices)

    if "fraud_type" in frame.columns:
        # Per-typology recall counts only positives, so it uses the dedicated
        # helper rather than the FPR/FNR subgroup table.
        report["recall_by_fraud_type"] = recall_by_group(
            probabilities, labels, frame["fraud_type"], threshold
        ).to_dict(orient="records")

    report["baseline_costs"] = baseline_costs(labels, config.cost)

    return report


def evaluate(
    model: RiskModel,
    frame: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Score a labelled frame with an existing model and report on it.

    The regression-test entrypoint: run a frozen set of applications through a
    candidate model and compare against the incumbent before promoting it.
    """
    config = config or model.config

    if LABEL not in frame.columns:
        raise KeyError(f"frame has no {LABEL!r} column")

    probabilities = model.risk_score(frame)
    labels = frame[LABEL].to_numpy()

    report = _build_report(probabilities, frame, model.thresholds.accept_below, config)
    report["decisions"] = decision_report(
        probabilities, labels, model.thresholds, config.cost
    ).to_dict(orient="records")

    bands = decision_report(probabilities, labels, model.thresholds, config.cost)
    report["expected_cost"] = bands.attrs["expected_cost"]
    report["cost_per_application"] = bands.attrs["cost_per_application"]

    return report


def summarise(report: dict[str, Any]) -> str:
    """Human-readable digest of a report, for the CLI and for logs."""
    overall = report["overall"]
    lines = [
        f"applications      : {overall['n']:,}",
        f"fraud rate        : {overall['base_rate']:.2%} ({overall['n_positive']:,} cases)",
        f"PR-AUC            : {overall['pr_auc']:.4f}",
        f"ROC-AUC           : {overall['roc_auc']:.4f}",
        f"Brier             : {overall['brier']:.5f}",
        f"calibration (ECE) : {overall['ece']:.5f}",
    ]

    for key, value in overall.items():
        if key.startswith("recall_at_fpr_"):
            lines.append(f"recall @ FPR {key.rsplit('_', 1)[-1]:<6}: {value:.4f}")

    iso = report.get("iso_30107_3")
    if iso:
        lines.append(
            f"ISO 30107-3       : APCER {iso['apcer']:.4f} / BPCER {iso['bpcer']:.4f} "
            f"@ threshold {iso['threshold']:.4f}"
        )

    for key, value in report.items():
        if key.startswith("slice_"):
            lines.append(
                f"inequity rate ({key.removeprefix('slice_')}): {value['inequity_rate']:.3f}"
            )

    if "recall_by_fraud_type" in report:
        lines.append("recall by fraud type:")
        for row in report["recall_by_fraud_type"]:
            if row["n_positive"]:
                lines.append(f"  {row['group']:<18} {row['recall']:.3f}  (n={row['n_positive']:,})")

    if "expected_cost" in report:
        lines.append(f"cost per application: {report['cost_per_application']:.2f}")
        for name, value in report["baseline_costs"].items():
            per_app = value / overall["n"]
            lines.append(f"  baseline {name:<12}: {per_app:.2f}")

    return "\n".join(lines)
