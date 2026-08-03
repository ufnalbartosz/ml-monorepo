"""The signal contract: every feature the risk engine consumes.

This module is the single source of truth for what a scoring request looks
like. The trainer derives its monotone constraints from it, the feature builder
derives its column order and dtypes from it, and the serving path validates
against it. If a signal is not declared here, it cannot reach the model.

**Monotone constraints are the interesting part.** XGBoost can be told that its
output must be non-decreasing (`+1`) or non-increasing (`-1`) in a given
feature. Where the direction is genuinely unambiguous, constraining it buys
three things: robustness when an attacker shifts the input distribution,
explanations that hold up in an audit, and a model that cannot learn an
artefact of the training sample that reverses in production.

The constraints below are deliberately sparse. A wrong constraint is worse than
no constraint, and two signals that *look* obviously monotone are not:

* ``face_match_score`` — higher similarity looks like lower risk, and for a
  careless impostor it is. But an injection or deepfake attack **optimises for
  a high match score**; that is the whole point of it. Constraining risk to
  fall as match rises would hand the attacker a guaranteed direction to push.
* ``doc_ocr_confidence`` — a cleanly forged document reads *better* than a
  genuine dog-eared one.

Both are left unconstrained so the model can learn the interaction with the
attack signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    """Which part of the verification flow produced a signal.

    Grouping by stage is what makes "which signals would have caught this
    fraud?" answerable, and it is how the ablation tests are organised.
    """

    DOCUMENT = "document"
    BIOMETRIC = "biometric"
    PAD = "pad"
    INJECTION = "injection"
    DEVICE = "device"
    GRAPH = "graph"
    BEHAVIOURAL = "behavioural"


class Direction(StrEnum):
    """Constrained direction of a signal's effect on P(fraud)."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    UNCONSTRAINED = "unconstrained"

    @property
    def xgboost_constraint(self) -> int:
        """XGBoost's encoding: +1 non-decreasing, -1 non-increasing, 0 free."""
        return {
            Direction.INCREASES_RISK: 1,
            Direction.DECREASES_RISK: -1,
            Direction.UNCONSTRAINED: 0,
        }[self]


@dataclass(frozen=True)
class Signal:
    """One model input.

    :param optional: whether the signal can legitimately be absent at scoring
        time. NFC is the motivating case — many phones cannot read a chip, and
        many documents do not have one. Absence is encoded as NaN and left for
        XGBoost to route, which is a substantive reason to prefer a tree
        ensemble here over a model that needs imputation.
    :param categories: for categorical signals, the closed set of levels. Any
        value outside it is folded into :data:`UNKNOWN_CATEGORY` by the feature
        builder — XGBoost raises on an unseen category at predict time, so a
        new device OS would otherwise take scoring down.
    """

    name: str
    stage: Stage
    dtype: str
    description: str
    direction: Direction = Direction.UNCONSTRAINED
    optional: bool = False
    categories: tuple[str, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None

    @property
    def is_categorical(self) -> bool:
        return self.categories is not None


#: Level used for any category not seen during training.
UNKNOWN_CATEGORY = "__unknown__"

_UNIT = {"minimum": 0.0, "maximum": 1.0}

SIGNALS: tuple[Signal, ...] = (
    # ---- Document -------------------------------------------------------
    Signal(
        "doc_ocr_confidence",
        Stage.DOCUMENT,
        "float",
        "Mean OCR field confidence. Unconstrained: a clean forgery reads better than a worn genuine document.",
        **_UNIT,
    ),
    Signal(
        "doc_mrz_checksum_valid",
        Stage.DOCUMENT,
        "float",
        "MRZ check digits agree. A deterministic arithmetic check, so the direction is unambiguous.",
        direction=Direction.DECREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "doc_template_match_score",
        Stage.DOCUMENT,
        "float",
        "Agreement with the known layout for this document type and issue year.",
        **_UNIT,
    ),
    Signal(
        "doc_tamper_score",
        Stage.DOCUMENT,
        "float",
        "Digital tampering likelihood: copy-move, splicing, font and print anomalies.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "doc_print_recapture_score",
        Stage.DOCUMENT,
        "float",
        "Likelihood the 'document' is a photo of a screen or a printout.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "nfc_chip_read",
        Stage.DOCUMENT,
        "float",
        "The eMRTD chip was read. Optional: most documents and many handsets cannot.",
        optional=True,
        **_UNIT,
    ),
    Signal(
        "nfc_passive_auth_passed",
        Stage.DOCUMENT,
        "float",
        "Chip data verified against the issuing country's signing certificate. "
        "The strongest single signal available: a cryptographic proof, not an inference from pixels.",
        direction=Direction.DECREASES_RISK,
        optional=True,
        **_UNIT,
    ),
    # ---- Biometric ------------------------------------------------------
    Signal(
        "face_match_score",
        Stage.BIOMETRIC,
        "float",
        "Cosine similarity, document portrait against live capture. "
        "Unconstrained on purpose: injection attacks optimise this upwards.",
        **_UNIT,
    ),
    Signal(
        "face_quality_selfie",
        Stage.BIOMETRIC,
        "float",
        "Capture quality of the live image (ISO/IEC 29794-5 style).",
        **_UNIT,
    ),
    Signal(
        "face_quality_document",
        Stage.BIOMETRIC,
        "float",
        "Capture quality of the document portrait.",
        **_UNIT,
    ),
    # ---- Presentation attack detection ----------------------------------
    Signal(
        "pad_attack_score",
        Stage.PAD,
        "float",
        "Presentation attack likelihood: print, replay, mask. ISO/IEC 30107-3 subsystem output.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "pad_active_challenge_passed",
        Stage.PAD,
        "float",
        "Randomised challenge (illumination sequence) satisfied. Optional: not every flow issues one.",
        direction=Direction.DECREASES_RISK,
        optional=True,
        **_UNIT,
    ),
    Signal(
        "rppg_pulse_detected",
        Stage.PAD,
        "float",
        "A plausible pulse waveform was recovered from skin colour change. "
        "Optional: needs enough frames and stable lighting.",
        optional=True,
        **_UNIT,
    ),
    # ---- Injection --------------------------------------------------
    Signal(
        "device_attestation_passed",
        Stage.INJECTION,
        "float",
        "Play Integrity / App Attest verdict: a genuine unmodified app on a genuine device.",
        direction=Direction.DECREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "virtual_camera_detected",
        Stage.INJECTION,
        "float",
        "A virtual camera driver or stream hook was present.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "sdk_integrity_failed",
        Stage.INJECTION,
        "float",
        "Client SDK tamper or hooking detection fired.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "frame_timing_anomaly_score",
        Stage.INJECTION,
        "float",
        "Inter-frame timing inconsistent with a live camera pipeline.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    # ---- Device and network ---------------------------------------------
    Signal(
        "device_is_emulator",
        Stage.DEVICE,
        "float",
        "The session ran on an emulator or virtual machine.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "ip_is_hosting",
        Stage.DEVICE,
        "float",
        "Source IP belongs to a hosting provider, VPN or proxy.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "ip_country_mismatch",
        Stage.DEVICE,
        "float",
        "IP geolocation disagrees with the document's issuing country.",
        **_UNIT,
    ),
    Signal(
        "device_os",
        Stage.DEVICE,
        "category",
        "Client platform. Categorical, with a closed level set so an unseen value cannot break scoring.",
        categories=("android", "ios", "web", UNKNOWN_CATEGORY),
    ),
    # ---- Velocity and graph ---------------------------------------------
    Signal(
        "applications_per_device_7d",
        Stage.GRAPH,
        "float",
        "Distinct applications from this device in the last 7 days.",
        direction=Direction.INCREASES_RISK,
        minimum=0.0,
    ),
    Signal(
        "applications_per_ip_7d",
        Stage.GRAPH,
        "float",
        "Distinct applications from this IP in the last 7 days.",
        direction=Direction.INCREASES_RISK,
        minimum=0.0,
    ),
    Signal(
        "graph_ring_size",
        Stage.GRAPH,
        "float",
        "Size of the connected component this applicant sits in, over shared device/IP/PII edges. "
        "This is what catches an organised ring whose individual applications all look clean.",
        direction=Direction.INCREASES_RISK,
        minimum=0.0,
    ),
    Signal(
        "shared_pii_count",
        Stage.GRAPH,
        "float",
        "Other applicants sharing a phone number, address or email root.",
        direction=Direction.INCREASES_RISK,
        minimum=0.0,
    ),
    # ---- Behavioural ----------------------------------------------------
    Signal(
        "session_duration_s",
        Stage.BEHAVIOURAL,
        "float",
        "Wall-clock duration of the onboarding session. Unconstrained: both unusually fast "
        "(scripted) and unusually slow (coached) sessions carry risk.",
        minimum=0.0,
    ),
    Signal(
        "form_paste_ratio",
        Stage.BEHAVIOURAL,
        "float",
        "Fraction of form fields filled by paste rather than keystrokes.",
        direction=Direction.INCREASES_RISK,
        **_UNIT,
    ),
    Signal(
        "retry_count",
        Stage.BEHAVIOURAL,
        "float",
        "Capture retries in this session.",
        direction=Direction.INCREASES_RISK,
        minimum=0.0,
    ),
)

#: Model input order. Fixed, because XGBoost matches features by position when
#: handed a bare array and the serving path must not reorder them.
FEATURE_NAMES: tuple[str, ...] = tuple(signal.name for signal in SIGNALS)

SIGNALS_BY_NAME: dict[str, Signal] = {signal.name: signal for signal in SIGNALS}

#: Column holding the binary fraud outcome.
LABEL = "is_fraud"

#: Columns that identify an application but must never be model inputs.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "application_id",
    "applicant_id",
    "device_id",
    "timestamp",
)

#: Columns used only to slice evaluation. Feeding these to the model would be
#: both a fairness problem and, for fraud_type, target leakage.
SLICE_COLUMNS: tuple[str, ...] = ("age_band", "skin_tone_band", "region", "fraud_type")


def signals_for_stage(stage: Stage) -> tuple[Signal, ...]:
    return tuple(signal for signal in SIGNALS if signal.stage is stage)


def feature_names_for_stages(*stages: Stage) -> tuple[str, ...]:
    """Feature names belonging to the given stages, in model order.

    Used by the ablation tests, which ask what a model built on only part of
    the stack can and cannot catch.
    """
    wanted = set(stages)
    return tuple(signal.name for signal in SIGNALS if signal.stage in wanted)


def monotone_constraints(feature_names: tuple[str, ...] = FEATURE_NAMES) -> dict[str, int]:
    """XGBoost ``monotone_constraints`` for the given features."""
    return {name: SIGNALS_BY_NAME[name].direction.xgboost_constraint for name in feature_names}


def categorical_names(feature_names: tuple[str, ...] = FEATURE_NAMES) -> tuple[str, ...]:
    return tuple(name for name in feature_names if SIGNALS_BY_NAME[name].is_categorical)


def optional_names(feature_names: tuple[str, ...] = FEATURE_NAMES) -> tuple[str, ...]:
    return tuple(name for name in feature_names if SIGNALS_BY_NAME[name].optional)
