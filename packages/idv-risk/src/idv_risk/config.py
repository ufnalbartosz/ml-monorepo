"""One typed, serialisable config object.

Anything that changes the model lives here and gets written next to the
artifact. A hyper-parameter that exists only as a literal in the training
script is a hyper-parameter you cannot reproduce, and "which settings produced
the model currently rejecting 3% of applicants?" is a question that gets asked
under time pressure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """XGBoost hyper-parameters.

    The defaults are deliberately conservative rather than tuned: shallow
    trees, a low learning rate and early stopping. On a fraud table with a few
    thousand positives, a deep unregularised ensemble will happily memorise the
    training rings and report an excellent score that does not survive contact
    with next month's traffic.
    """

    n_estimators: int = 600
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 2.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    #: Stop when the validation metric has not improved for this many rounds.
    early_stopping_rounds: int = 50
    #: Area under the precision-recall curve. The right early-stopping metric
    #: under heavy imbalance: AUC-ROC is dominated by the vast negative class
    #: and barely moves when the model gets better at the positives.
    eval_metric: str = "aucpr"
    #: Weight positives by the class ratio. None means "compute it from the
    #: training labels"; set a float to override, or 1.0 to disable.
    scale_pos_weight: float | None = None
    #: Apply the monotone constraints declared in the schema.
    use_monotone_constraints: bool = True
    tree_method: str = "hist"
    random_state: int = 20260801
    n_jobs: int = 4


@dataclass(frozen=True)
class CalibrationConfig:
    """How raw scores become probabilities you can put a threshold on.

    ``isotonic`` is the default because the decision policy compares scores
    against absolute probability bands, and an uncalibrated boosted ensemble is
    badly overconfident — the bands would not mean what they say. Isotonic is
    non-parametric and can overfit on small calibration sets, hence
    ``min_samples``: below that, fall back to sigmoid (Platt), which has two
    parameters and cannot.
    """

    method: str = "isotonic"
    fallback_method: str = "sigmoid"
    min_samples: int = 1_000
    #: Minimum positives required before isotonic is trusted.
    min_positives: int = 50


@dataclass(frozen=True)
class CostConfig:
    """The business cost of each outcome, in whatever currency you use.

    These numbers, not accuracy, choose the operating point. A model tuned to
    maximise accuracy on a 4%-fraud table is tuned to say "genuine" and be
    right 96% of the time.

    The defaults encode a plausible consumer-onboarding shape: letting a
    fraudster through costs far more than annoying a genuine applicant, and a
    manual review sits in between — cheap enough to use often, expensive
    enough that you cannot review everything.
    """

    #: Fraud accepted: chargeback, remediation, regulatory exposure.
    false_accept: float = 500.0
    #: Genuine applicant rejected: lost lifetime value plus support contact.
    false_reject: float = 60.0
    #: Manual review, either outcome. Analyst time plus applicant friction.
    step_up: float = 12.0
    #: Residual loss when a review lets a fraudster through anyway.
    step_up_miss_rate: float = 0.10
    #: Genuine applicants who abandon rather than complete a step-up.
    step_up_abandon_rate: float = 0.15


@dataclass(frozen=True)
class TrainingConfig:
    """Everything needed to reproduce a training run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    valid_fraction: float = 0.15
    test_fraction: float = 0.15
    #: Entities purged from later periods. Applicant only, deliberately:
    #: adding "device_id" here purges the shared-device rows that *are* the
    #: synthetic-identity rings, deleting that typology from the evaluation
    #: set. See the :mod:`idv_risk.splits` docstring.
    group_columns: tuple[str, ...] = ("applicant_id",)
    time_column: str = "timestamp"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        return cls(
            model=ModelConfig(**payload.get("model", {})),
            calibration=CalibrationConfig(**payload.get("calibration", {})),
            cost=CostConfig(**payload.get("cost", {})),
            valid_fraction=payload.get("valid_fraction", 0.15),
            test_fraction=payload.get("test_fraction", 0.15),
            group_columns=tuple(payload.get("group_columns", ("applicant_id",))),
            time_column=payload.get("time_column", "timestamp"),
        )

    def replace(self, **changes: Any) -> TrainingConfig:
        return replace(self, **changes)
