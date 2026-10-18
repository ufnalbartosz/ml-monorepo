"""Evaluation that says something.

A single number cannot describe a fraud model, and the number people reach for
first — accuracy — is actively misleading at a 4% base rate, where predicting
"genuine" for everything scores 96%.

What this module reports instead, and why each earns its place:

* **PR-AUC (average precision)** as the headline. ROC-AUC is dominated by the
  enormous negative class and barely moves when the model gets better at the
  thing you care about.
* **Recall at a fixed false-positive rate.** The operational question is never
  "how good is the model" but "at the friction we can afford, how much fraud do
  we catch". Reported at several FPRs because the answer is a curve.
* **Calibration** (ECE, Brier), because the decision policy is arithmetic on
  probabilities.
* **Per-typology recall**, which is where fusion justifies itself: an aggregate
  recall of 0.85 can be 0.99 on presentation attacks and 0.10 on rings.
* **Subgroup differentials** using the ISO/IEC 19795-10:2024 measures. Error
  rates in biometric systems vary across demographic groups, and an aggregate
  number conceals exactly the failure that produces both harm and litigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from idv_risk.calibration import expected_calibration_error

DEFAULT_FPR_TARGETS: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)


@dataclass(frozen=True)
class ScoreReport:
    """Threshold-free quality of a set of scores."""

    n: int
    n_positive: int
    base_rate: float
    pr_auc: float
    roc_auc: float
    brier: float
    ece: float
    recall_at_fpr: dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "n": self.n,
            "n_positive": self.n_positive,
            "base_rate": self.base_rate,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "brier": self.brier,
            "ece": self.ece,
        }
        payload.update({f"recall_at_fpr_{k:g}": v for k, v in self.recall_at_fpr.items()})
        return payload


def recall_at_fpr(
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_fpr: float,
) -> float:
    """Recall at the highest threshold whose FPR does not exceed ``target_fpr``.

    "At a 1% false-positive rate, what share of fraud do we catch?" — the
    question a fraud lead actually asks, because the FPR is the friction budget
    and it is fixed by the business, not by the model.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1)")

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(labels, probabilities)
    allowed = fpr <= target_fpr

    if not allowed.any():
        return 0.0

    return float(tpr[allowed].max())


def score_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    fpr_targets: tuple[float, ...] = DEFAULT_FPR_TARGETS,
) -> ScoreReport:
    """Threshold-free metrics for one set of scores."""
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    if len(probabilities) == 0:
        raise ValueError("cannot score an empty population")
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels disagree on length: {len(probabilities)} vs {len(labels)}"
        )

    n_positive = int(labels.sum())
    single_class = len(np.unique(labels)) < 2

    return ScoreReport(
        n=len(labels),
        n_positive=n_positive,
        base_rate=float(labels.mean()),
        pr_auc=float("nan")
        if single_class
        else float(average_precision_score(labels, probabilities)),
        roc_auc=float("nan") if single_class else float(roc_auc_score(labels, probabilities)),
        brier=float(brier_score_loss(labels, probabilities)),
        ece=expected_calibration_error(probabilities, labels),
        recall_at_fpr={
            target: recall_at_fpr(probabilities, labels, target) for target in fpr_targets
        },
    )


def recall_by_group(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Detection rate per group at a fixed threshold.

    Used for per-typology recall — "which attacks does this model actually
    stop?" — where the group is the fraud type and only positives count.
    """
    flagged = np.asarray(probabilities) >= threshold
    labels = np.asarray(labels)
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for name, index in groups.groupby(groups, observed=True).groups.items():
        positions = np.asarray(index)
        positives = labels[positions] == 1
        n_positive = int(positives.sum())
        rows.append(
            {
                "group": name,
                "n": len(positions),
                "n_positive": n_positive,
                "recall": float(flagged[positions][positives].mean()) if n_positive else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def subgroup_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: pd.Series,
    threshold: float,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """False-positive and false-negative rate per demographic group.

    In biometric terms these are the false-match and false-non-match rates that
    ISO/IEC 19795-10 asks to be reported per group rather than pooled.

    ``min_group_size`` applies to the genuine and fraud counts *separately*, not
    to the group total: estimating an FNR needs fraud cases, and at a 4% base
    rate a group of 500 holds only 20 of them. Groups below the bar are still
    reported — you should see them — but :func:`inequity_rate` leaves them out.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")
    groups = pd.Series(groups).reset_index(drop=True)

    flagged = probabilities >= threshold

    rows = []
    for name, index in groups.groupby(groups, observed=True).groups.items():
        positions = np.asarray(index)
        group_labels = labels[positions]
        group_flagged = flagged[positions]

        genuine = group_labels == 0
        fraud = group_labels == 1

        n_genuine = int(genuine.sum())
        n_fraud = int(fraud.sum())

        rows.append(
            {
                "group": name,
                "n": len(positions),
                "n_genuine": n_genuine,
                "n_fraud": n_fraud,
                # False positive: a genuine applicant flagged as fraud.
                "fpr": float(group_flagged[genuine].mean()) if n_genuine else np.nan,
                # False negative: fraud that got through.
                "fnr": float((~group_flagged[fraud]).mean()) if n_fraud else np.nan,
                # Jeffreys-smoothed rates. The raw rate of a group that happened
                # to make zero errors is 0.0, and a ratio against it is infinite
                # (or, with a fixed epsilon floor, an arbitrary number set by the
                # epsilon). Smoothing by (errors + 1/2) / (n + 1) keeps the
                # estimate bounded and shrinks it toward the middle in
                # proportion to how little data the group has, which is exactly
                # the uncertainty being represented.
                "fpr_smoothed": _jeffreys(int(group_flagged[genuine].sum()), n_genuine),
                "fnr_smoothed": _jeffreys(int((~group_flagged[fraud]).sum()), n_fraud),
                "sufficient_data": min(n_genuine, n_fraud) >= min_group_size,
            }
        )

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def _jeffreys(events: int, trials: int) -> float:
    """Jeffreys interval point estimate: (events + 1/2) / (trials + 1)."""
    if trials <= 0:
        return float("nan")
    return (events + 0.5) / (trials + 1.0)


def inequity_rate(subgroups: pd.DataFrame) -> float:
    """Inequity Rate (IR) over demographic groups, per ISO/IEC 19795-10.

    The product of the max-to-min ratios of the two error rates across groups::

        IR = (max FPR / min FPR) x (max FNR / min FNR)

    ``IR = 1`` means every group sees identical error rates. It rises fast: a
    group with double the false-positive rate *and* double the false-negative
    rate of the best group gives IR = 4.

    Two guards against reporting noise as unfairness, which is the failure mode
    that discredits a fairness metric the first time someone acts on it:

    * only groups with at least ``min_group_size`` genuine *and* fraud cases
      count — the constraint is on each denominator, not on the group total,
      because a 500-person group holding two fraud cases still cannot estimate
      an FNR;
    * the smoothed rates are used, so a group that made zero errors contributes
      a finite ratio rather than an infinite one.

    Returns NaN when fewer than two groups qualify, which is the honest answer:
    with one group there is no differential to measure.
    """
    usable = (
        subgroups[subgroups["sufficient_data"]] if "sufficient_data" in subgroups else subgroups
    )
    usable = usable.dropna(subset=["fpr_smoothed", "fnr_smoothed"])

    if len(usable) < 2:
        return float("nan")

    fpr = usable["fpr_smoothed"].to_numpy()
    fnr = usable["fnr_smoothed"].to_numpy()

    return float((fpr.max() / fpr.min()) * (fnr.max() / fnr.min()))


def apcer_bpcer(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """ISO/IEC 30107-3 error rates, in that standard's vocabulary.

    * **APCER** — attack presentations classified as bona fide. The security
      failure: fraud that got through.
    * **BPCER** — bona fide presentations classified as attacks. The conversion
      failure: genuine applicants rejected.

    Strictly these are defined for a PAD subsystem rather than a fusion engine,
    but reporting the fusion decision in the same vocabulary is what lets it be
    compared against the PAD component and discussed with an assessor who
    thinks in those terms. Always quote them as a pair at a stated threshold —
    either alone is meaningless.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    attacks = labels == 1
    bona_fide = labels == 0

    if not attacks.any() or not bona_fide.any():
        raise ValueError("need both attack and bona fide presentations to compute APCER/BPCER")

    flagged = probabilities >= threshold

    apcer = float((~flagged[attacks]).mean())
    bpcer = float(flagged[bona_fide].mean())

    return apcer, bpcer


def evaluation_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    slices: dict[str, pd.Series] | None = None,
    fpr_targets: tuple[float, ...] = DEFAULT_FPR_TARGETS,
) -> dict:
    """Everything above, assembled into one nested dict for logging."""
    summary: dict = {"overall": score_report(probabilities, labels, fpr_targets).to_dict()}

    apcer, bpcer = apcer_bpcer(probabilities, labels, threshold)
    summary["iso_30107_3"] = {"threshold": threshold, "apcer": apcer, "bpcer": bpcer}

    for name, values in (slices or {}).items():
        table = subgroup_report(probabilities, labels, values, threshold)
        summary[f"slice_{name}"] = {
            "groups": table.to_dict(orient="records"),
            "inequity_rate": inequity_rate(table),
        }

    return summary
