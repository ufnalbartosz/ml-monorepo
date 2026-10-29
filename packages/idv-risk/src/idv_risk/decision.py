"""Turning a probability into accept / step-up / reject.

Two thresholds, not one. A binary accept/reject engine forces a single
compromise between fraud loss and lost customers; a middle band lets the
uncertain applications go to a stronger check — NFC, a video call, manual
review — and that band is where a risk-based system earns its keep.

**The thresholds come from costs, not from accuracy.** On a table that is 96%
genuine, the accuracy-maximising policy is to accept everything. The policy
here instead minimises expected cost over a grid of candidate boundaries,
given the four numbers in :class:`~idv_risk.config.CostConfig`. Two of those
numbers make the step-up band honest rather than free:

* ``step_up_miss_rate`` — reviews are not perfect; some fraud survives one.
* ``step_up_abandon_rate`` — genuine applicants drop out when asked for more,
  and an abandoned good customer costs the same as a rejected one.

Without those, the optimiser would route everything to step-up and report a
wonderful expected cost that no operations team could staff.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from idv_risk.config import CostConfig


class Decision(StrEnum):
    ACCEPT = "accept"
    STEP_UP = "step_up"
    REJECT = "reject"


@dataclass(frozen=True)
class Thresholds:
    """Boundaries of the three risk bands, on the calibrated probability."""

    accept_below: float
    reject_at_or_above: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.accept_below <= 1.0:
            raise ValueError("accept_below must be in [0, 1]")
        if not 0.0 <= self.reject_at_or_above <= 1.0:
            raise ValueError("reject_at_or_above must be in [0, 1]")
        if self.accept_below > self.reject_at_or_above:
            raise ValueError(
                f"accept_below ({self.accept_below}) must not exceed "
                f"reject_at_or_above ({self.reject_at_or_above})"
            )

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        """Map probabilities to decision labels."""
        probabilities = np.asarray(probabilities, dtype="float64")

        decisions = np.full(len(probabilities), Decision.STEP_UP.value, dtype=object)
        decisions[probabilities < self.accept_below] = Decision.ACCEPT.value
        decisions[probabilities >= self.reject_at_or_above] = Decision.REJECT.value

        return decisions

    def to_dict(self) -> dict[str, float]:
        return {
            "accept_below": self.accept_below,
            "reject_at_or_above": self.reject_at_or_above,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> Thresholds:
        return cls(
            accept_below=float(payload["accept_below"]),
            reject_at_or_above=float(payload["reject_at_or_above"]),
        )


def expected_cost(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Thresholds,
    cost: CostConfig,
) -> float:
    """Total expected cost of applying ``thresholds`` to this population.

    Per application:

    * accepted fraud → ``false_accept``
    * accepted genuine → 0
    * rejected fraud → 0 (the loss was avoided)
    * rejected genuine → ``false_reject``
    * stepped-up fraud → ``step_up`` + ``step_up_miss_rate`` x ``false_accept``
    * stepped-up genuine → ``step_up`` + ``step_up_abandon_rate`` x ``false_reject``
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels disagree on length: {len(probabilities)} vs {len(labels)}"
        )
    if len(probabilities) == 0:
        raise ValueError("cannot evaluate cost over an empty population")

    decisions = thresholds.decide(probabilities)
    is_fraud = labels == 1

    accepted = decisions == Decision.ACCEPT.value
    rejected = decisions == Decision.REJECT.value
    stepped = decisions == Decision.STEP_UP.value

    total = 0.0
    total += float((accepted & is_fraud).sum()) * cost.false_accept
    total += float((rejected & ~is_fraud).sum()) * cost.false_reject
    total += float(stepped.sum()) * cost.step_up
    total += float((stepped & is_fraud).sum()) * cost.step_up_miss_rate * cost.false_accept
    total += float((stepped & ~is_fraud).sum()) * cost.step_up_abandon_rate * cost.false_reject

    return total


def optimise_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    cost: CostConfig,
    n_candidates: int = 100,
    max_step_up_rate: float | None = 0.20,
) -> Thresholds:
    """Grid-search the boundary pair that minimises expected cost.

    :param max_step_up_rate: hard cap on the fraction routed to review, or
        ``None`` for uncapped. This is an operational constraint, not a
        statistical one — a policy that reviews 40% of traffic may well be
        cost-optimal on paper and still be undeployable because the review team
        does not exist. Capping it here keeps the optimiser inside what can
        actually be staffed.

    Candidates are drawn from the observed score quantiles rather than a
    uniform grid, so the search spends its resolution where the scores are.
    """
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    if len(probabilities) == 0:
        raise ValueError("cannot optimise thresholds over an empty population")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2")

    quantiles = np.linspace(0.0, 1.0, n_candidates)
    candidates = np.unique(np.quantile(probabilities, quantiles))
    # Allow a band to be empty: 1.0 as the reject boundary means "never
    # reject outright", which is the right answer when false_reject dominates.
    candidates = np.unique(np.concatenate([candidates, [0.0, 1.0]]))

    best: Thresholds | None = None
    best_cost = np.inf

    for accept_below in candidates:
        for reject_at in candidates[candidates >= accept_below]:
            thresholds = Thresholds(float(accept_below), float(reject_at))

            if max_step_up_rate is not None:
                step_up_rate = float(
                    np.mean(thresholds.decide(probabilities) == Decision.STEP_UP.value)
                )
                if step_up_rate > max_step_up_rate:
                    continue

            candidate_cost = expected_cost(probabilities, labels, thresholds, cost)
            if candidate_cost < best_cost:
                best_cost, best = candidate_cost, thresholds

    if best is None:
        # Only reachable if every candidate pair breaches the step-up cap.
        raise ValueError(
            f"no threshold pair satisfies max_step_up_rate={max_step_up_rate}; "
            "raise the cap or pass None"
        )

    return best


def decision_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: Thresholds,
    cost: CostConfig | None = None,
) -> pd.DataFrame:
    """Per-band volume and fraud rate.

    The table an operations team actually reads: how much traffic lands in each
    band, and how much of it is fraud. A step-up band whose fraud rate is
    indistinguishable from the accept band is a band that is buying friction
    and no security.
    """
    cost = cost or CostConfig()
    probabilities = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(labels, dtype="float64")

    decisions = thresholds.decide(probabilities)
    total = len(probabilities)

    rows = []
    for decision in (Decision.ACCEPT, Decision.STEP_UP, Decision.REJECT):
        in_band = decisions == decision.value
        count = int(in_band.sum())
        rows.append(
            {
                "decision": decision.value,
                "count": count,
                "share": count / total if total else 0.0,
                "fraud_count": int(labels[in_band].sum()),
                "fraud_rate": float(labels[in_band].mean()) if count else np.nan,
            }
        )

    report = pd.DataFrame(rows)
    report.attrs["expected_cost"] = expected_cost(probabilities, labels, thresholds, cost)
    report.attrs["cost_per_application"] = report.attrs["expected_cost"] / total

    return report


def baseline_costs(
    labels: np.ndarray,
    cost: CostConfig,
) -> dict[str, float]:
    """Cost of the two trivial policies, for comparison.

    A model is only worth deploying if it beats "accept everything" and "review
    everything". Reporting those two alongside the optimised policy keeps that
    honest — it is a surprisingly common outcome that a model with a fine
    PR-AUC saves nothing once the review cost is counted.
    """
    labels = np.asarray(labels, dtype="float64")
    n = len(labels)
    if n == 0:
        raise ValueError("cannot evaluate baselines over an empty population")

    fraud = float(labels.sum())
    genuine = n - fraud

    return {
        "accept_all": fraud * cost.false_accept,
        "reject_all": genuine * cost.false_reject,
        "step_up_all": (
            n * cost.step_up
            + fraud * cost.step_up_miss_rate * cost.false_accept
            + genuine * cost.step_up_abandon_rate * cost.false_reject
        ),
    }
