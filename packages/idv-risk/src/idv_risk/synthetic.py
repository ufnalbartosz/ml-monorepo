"""A synthetic identity-verification data-set.

Real IDV data is biometric data: special-category under GDPR Article 9, and in
Illinois it carries a private right of action. It cannot be committed to a
repository, and a package whose tests need it is a package nobody can run. So
the generator here is the reference data-set, and a real deployment swaps it
for :func:`idv_risk.features.build_features` over its own warehouse table.

The generator is not decoration. It encodes the claim the whole package rests
on — **no single signal family catches everything** — by giving each fraud
typology a different blind spot:

===================== ==================================================
typology              what it defeats
===================== ==================================================
``presentation``      Nothing clever; PAD sees it. The easy case.
``injection``         PAD sees a *perfect* face: high match, no artefacts,
                      because the frames never met a camera. Only device
                      attestation and frame timing fire.
``document_forgery``  Face and liveness are a real person — the applicant
                      is real, the document is not. Only the document
                      stage and the absent NFC chip fire.
``synthetic_id``      Every per-session signal is clean, because every
                      session *is* clean. Only velocity and graph
                      features fire. This is the one a pure biometrics
                      stack cannot see at all.
===================== ==================================================

``test_ablations.py`` turns that table into assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from idv_risk.schema import (
    FEATURE_NAMES,
    IDENTIFIER_COLUMNS,
    LABEL,
    SIGNALS_BY_NAME,
    UNKNOWN_CATEGORY,
)


class FraudType(StrEnum):
    GENUINE = "genuine"
    PRESENTATION = "presentation"
    INJECTION = "injection"
    DOCUMENT_FORGERY = "document_forgery"
    SYNTHETIC_ID = "synthetic_id"


ATTACK_TYPES: tuple[FraudType, ...] = tuple(t for t in FraudType if t is not FraudType.GENUINE)


@dataclass(frozen=True)
class GeneratorConfig:
    """Shape of the generated population.

    Defaults give roughly a 4% fraud rate, which is high for consumer
    onboarding and low for a crypto exchange. The point of the parameter is
    that the imbalance handling has to be exercised at whatever rate you pick.
    """

    n_applications: int = 12_000
    fraud_rate: float = 0.04
    #: Relative mix of the four typologies among fraudulent applications.
    typology_mix: dict[FraudType, float] = field(
        default_factory=lambda: {
            FraudType.PRESENTATION: 0.30,
            FraudType.INJECTION: 0.30,
            FraudType.DOCUMENT_FORGERY: 0.25,
            FraudType.SYNTHETIC_ID: 0.15,
        }
    )
    #: Fraction of sessions where the handset could not read an NFC chip.
    nfc_unavailable_rate: float = 0.55
    #: Fraction of sessions with too few frames for a pulse estimate.
    rppg_unavailable_rate: float = 0.30
    #: Fraction of flows that issue an active challenge.
    active_challenge_rate: float = 0.40
    #: Fraction of attacks that invest in suppressing their own tells. Drives
    #: how much class overlap there is, and so how hard the problem is.
    sophisticated_attack_rate: float = 0.45
    #: Fraction of genuine applicants who trip an attack signal innocently.
    unlucky_genuine_rate: float = 0.06
    #: Fraction of applications that are a repeat by an earlier applicant.
    applicant_repeat_rate: float = 0.08
    #: Fraction that reuse an earlier device (households, shared computers).
    device_reuse_rate: float = 0.15
    #: How many rows back a repeat may reach. Rows are in time order, so this
    #: is a recency window, and keeping it short is what makes repeats look
    #: like retries rather than unrelated re-applications a quarter later.
    repeat_window: int = 120
    #: Days spanned by the generated timestamps.
    days: int = 120
    seed: int = 20260801


#: Numeric model signals, in schema order. ``device_os`` is excluded because
#: it is categorical and cannot be interpolated.
_NUMERIC_SIGNALS: tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if not SIGNALS_BY_NAME[name].is_categorical
)

AGE_BANDS = ("18-29", "30-44", "45-59", "60+")
# Coarse Fitzpatrick groupings, the axis NIST FRVT Part 3 found the largest
# differentials along. Present so fairness slicing can be tested, never as a
# model input.
SKIN_TONE_BANDS = ("I-II", "III-IV", "V-VI")
REGIONS = ("EU", "NA", "LATAM", "APAC")
DEVICE_OS = ("android", "ios", "web")


def _local_repeats(
    rng: np.random.Generator,
    n: int,
    repeat_rate: float,
    window: int,
    prefix: str,
) -> list[str]:
    """Identifiers where a repeat reuses one from a nearby earlier row.

    Rows are already in time order, so "nearby earlier row" means "recently",
    which is how retries and shared household devices actually look.
    """
    ids = np.arange(n)

    repeats = np.flatnonzero(rng.random(n) < repeat_rate)
    for position in repeats:
        if position == 0:
            continue
        back = int(rng.integers(1, window + 1))
        ids[position] = ids[max(0, position - back)]

    return [f"{prefix}_{i:07d}" for i in ids]


def _beta(rng: np.random.Generator, a: float, b: float, size: int) -> np.ndarray:
    return rng.beta(a, b, size=size).astype(np.float64)


def _bernoulli(rng: np.random.Generator, p: float | np.ndarray, size: int) -> np.ndarray:
    return (rng.random(size) < p).astype(np.float64)


def generate(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Generate a labelled application table.

    The frame carries the model signals, the identifier columns, the slice
    columns and the label. It is deliberately *not* feature-engineered — that
    is :mod:`idv_risk.features`' job, and keeping the two apart is what lets
    the feature builder be tested against a realistic input.
    """
    config = config or GeneratorConfig()

    if not 0.0 < config.fraud_rate < 1.0:
        raise ValueError("fraud_rate must be in (0, 1)")
    if config.n_applications < 1:
        raise ValueError("n_applications must be positive")

    rng = np.random.default_rng(config.seed)
    n = config.n_applications

    fraud_type = _assign_typologies(rng, n, config)
    is_fraud = (fraud_type != FraudType.GENUINE.value).astype(int)

    frame = pd.DataFrame(
        {
            "application_id": [f"app_{i:07d}" for i in range(n)],
            "fraud_type": fraud_type,
            LABEL: is_fraud,
        }
    )

    _add_population(frame, rng, config)
    _add_genuine_baseline(frame, rng, config)

    # Snapshot the honest values before the attacks overwrite them. Blending
    # back toward this per-row baseline is what makes a "sophisticated" attack
    # sophisticated across *every* signal its typology touches, rather than
    # only the ones someone remembered to list.
    baseline = frame[list(_NUMERIC_SIGNALS)].copy()

    _apply_presentation_attacks(frame, rng)
    _apply_injection_attacks(frame, rng)
    _apply_document_forgery(frame, rng)
    _apply_synthetic_identity_rings(frame, rng)
    _apply_confusable_cases(frame, rng, config, baseline)
    _apply_missingness(frame, rng, config)

    return frame


def _assign_typologies(rng: np.random.Generator, n: int, config: GeneratorConfig) -> np.ndarray:
    total = sum(config.typology_mix.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"typology_mix must sum to 1.0, got {total}")

    labels = np.full(n, FraudType.GENUINE.value, dtype=object)
    fraud_mask = rng.random(n) < config.fraud_rate

    types = list(config.typology_mix)
    weights = np.array([config.typology_mix[t] for t in types])
    drawn = rng.choice([t.value for t in types], size=int(fraud_mask.sum()), p=weights)
    labels[fraud_mask] = drawn

    return labels


def _add_population(frame: pd.DataFrame, rng: np.random.Generator, config: GeneratorConfig) -> None:
    """Identifiers, timestamps and the demographic slice columns.

    Applications arrive in time order, and repeats are *temporally local*: a
    person who applies twice almost always does so within days, because it is
    a retry rather than an unrelated second life event. Scattering repeats
    uniformly across the whole window would be both unrealistic and quietly
    destructive — the split purges any applicant already seen in an earlier
    period, so uniformly-scattered repeats would purge most of the evaluation
    set rather than the few percent they should.
    """
    n = len(frame)

    start = np.datetime64("2026-01-01T00:00:00")
    offsets = np.sort(rng.integers(0, config.days * 24 * 3600, size=n)).astype("timedelta64[s]")
    frame["timestamp"] = pd.to_datetime(start + offsets)

    frame["applicant_id"] = _local_repeats(
        rng,
        n,
        repeat_rate=config.applicant_repeat_rate,
        window=config.repeat_window,
        prefix="person",
    )
    frame["device_id"] = _local_repeats(
        rng,
        n,
        repeat_rate=config.device_reuse_rate,
        window=config.repeat_window * 4,
        prefix="device",
    )

    frame["age_band"] = rng.choice(AGE_BANDS, size=n, p=[0.34, 0.32, 0.22, 0.12])
    frame["skin_tone_band"] = rng.choice(SKIN_TONE_BANDS, size=n, p=[0.45, 0.33, 0.22])
    frame["region"] = rng.choice(REGIONS, size=n, p=[0.40, 0.30, 0.15, 0.15])
    frame["device_os"] = rng.choice(DEVICE_OS, size=n, p=[0.48, 0.40, 0.12])


def _add_genuine_baseline(
    frame: pd.DataFrame, rng: np.random.Generator, config: GeneratorConfig
) -> None:
    """Signals as a legitimate applicant produces them.

    Attack typologies then overwrite the columns they affect, so anything an
    attack does not touch keeps a realistic honest value — which is what makes
    the blind spots in the table above real rather than assumed.
    """
    n = len(frame)

    frame["doc_ocr_confidence"] = 0.55 + 0.45 * _beta(rng, 6, 2, n)
    frame["doc_mrz_checksum_valid"] = _bernoulli(rng, 0.985, n)
    frame["doc_template_match_score"] = 0.5 + 0.5 * _beta(rng, 7, 2, n)
    frame["doc_tamper_score"] = 0.25 * _beta(rng, 1.5, 8, n)
    frame["doc_print_recapture_score"] = 0.25 * _beta(rng, 1.5, 8, n)
    frame["nfc_chip_read"] = _bernoulli(rng, 0.9, n)
    frame["nfc_passive_auth_passed"] = np.where(
        frame["nfc_chip_read"] > 0, _bernoulli(rng, 0.995, n), 0.0
    )

    # Capture quality drives match quality, and it is also the mechanism behind
    # demographic differentials: under-exposed captures score worse, and that
    # correlates with skin tone under poor lighting. Modelling it explicitly is
    # what gives the fairness metrics something real to find.
    quality_penalty = frame["skin_tone_band"].map({"I-II": 0.0, "III-IV": 0.02, "V-VI": 0.05})
    frame["face_quality_selfie"] = np.clip(
        0.55 + 0.45 * _beta(rng, 6, 2, n) - quality_penalty.to_numpy(), 0.0, 1.0
    )
    frame["face_quality_document"] = 0.45 + 0.5 * _beta(rng, 5, 2, n)
    frame["face_match_score"] = np.clip(
        0.62 + 0.35 * _beta(rng, 6, 2, n) - 0.12 * (1.0 - frame["face_quality_selfie"].to_numpy()),
        0.0,
        1.0,
    )

    frame["pad_attack_score"] = 0.20 * _beta(rng, 1.5, 9, n)
    frame["pad_active_challenge_passed"] = np.where(
        rng.random(n) < config.active_challenge_rate, _bernoulli(rng, 0.97, n), np.nan
    )
    frame["rppg_pulse_detected"] = _bernoulli(rng, 0.94, n)

    frame["device_attestation_passed"] = _bernoulli(rng, 0.97, n)
    frame["virtual_camera_detected"] = _bernoulli(rng, 0.004, n)
    frame["sdk_integrity_failed"] = _bernoulli(rng, 0.006, n)
    frame["frame_timing_anomaly_score"] = 0.20 * _beta(rng, 1.5, 9, n)

    frame["device_is_emulator"] = _bernoulli(rng, 0.006, n)
    frame["ip_is_hosting"] = _bernoulli(rng, 0.03, n)
    frame["ip_country_mismatch"] = _bernoulli(rng, 0.07, n)

    frame["applications_per_device_7d"] = rng.poisson(0.4, n).astype(float) + 1.0
    frame["applications_per_ip_7d"] = rng.poisson(0.8, n).astype(float) + 1.0
    frame["graph_ring_size"] = 1.0 + rng.poisson(0.2, n).astype(float)
    frame["shared_pii_count"] = rng.poisson(0.15, n).astype(float)

    frame["session_duration_s"] = np.clip(rng.lognormal(4.9, 0.45, n), 20.0, 1800.0)
    frame["form_paste_ratio"] = 0.3 * _beta(rng, 1.5, 6, n)
    frame["retry_count"] = rng.poisson(0.5, n).astype(float)


def _mask_for(frame: pd.DataFrame, typology: FraudType) -> np.ndarray:
    return (frame["fraud_type"] == typology.value).to_numpy()


def _apply_presentation_attacks(frame: pd.DataFrame, rng: np.random.Generator) -> None:
    """Print, replay or mask held up to a real camera. PAD is designed for this."""
    mask = _mask_for(frame, FraudType.PRESENTATION)
    k = int(mask.sum())
    if k == 0:
        return

    # Not every attack is caught: a well-made mask can score low. That overlap
    # is why PAD alone leaves residual risk.
    frame.loc[mask, "pad_attack_score"] = np.clip(0.35 + 0.6 * _beta(rng, 4, 2, k), 0, 1)
    frame.loc[mask, "doc_print_recapture_score"] = np.clip(0.2 + 0.6 * _beta(rng, 3, 3, k), 0, 1)
    frame.loc[mask, "rppg_pulse_detected"] = _bernoulli(rng, 0.15, k)
    frame.loc[mask, "face_quality_selfie"] = np.clip(0.3 + 0.4 * _beta(rng, 3, 3, k), 0, 1)
    frame.loc[mask, "face_match_score"] = np.clip(0.45 + 0.4 * _beta(rng, 4, 3, k), 0, 1)
    frame.loc[mask, "retry_count"] = rng.poisson(1.8, k).astype(float)
    challenge = frame.loc[mask, "pad_active_challenge_passed"].to_numpy()
    frame.loc[mask, "pad_active_challenge_passed"] = np.where(
        np.isnan(challenge), np.nan, _bernoulli(rng, 0.25, k)
    )


def _apply_injection_attacks(frame: pd.DataFrame, rng: np.random.Generator) -> None:
    """Frames fed straight into the stream, bypassing the camera.

    The defining property: **PAD sees nothing wrong**, because there is no
    presentation. The face is a high-quality deepfake that matches the document
    better than a real selfie does. Only the injection stage fires.
    """
    mask = _mask_for(frame, FraudType.INJECTION)
    k = int(mask.sum())
    if k == 0:
        return

    frame.loc[mask, "pad_attack_score"] = 0.18 * _beta(rng, 1.5, 9, k)
    frame.loc[mask, "rppg_pulse_detected"] = _bernoulli(rng, 0.75, k)
    # Synthesised media is *cleaner* than a phone selfie in a hallway.
    frame.loc[mask, "face_match_score"] = np.clip(0.80 + 0.19 * _beta(rng, 5, 2, k), 0, 1)
    frame.loc[mask, "face_quality_selfie"] = np.clip(0.80 + 0.19 * _beta(rng, 5, 2, k), 0, 1)

    frame.loc[mask, "device_attestation_passed"] = _bernoulli(rng, 0.30, k)
    frame.loc[mask, "virtual_camera_detected"] = _bernoulli(rng, 0.55, k)
    frame.loc[mask, "sdk_integrity_failed"] = _bernoulli(rng, 0.45, k)
    frame.loc[mask, "frame_timing_anomaly_score"] = np.clip(0.3 + 0.6 * _beta(rng, 3, 2, k), 0, 1)
    frame.loc[mask, "device_is_emulator"] = _bernoulli(rng, 0.35, k)
    frame.loc[mask, "ip_is_hosting"] = _bernoulli(rng, 0.45, k)
    frame.loc[mask, "device_os"] = rng.choice(["android", "web"], size=k, p=[0.4, 0.6])
    frame.loc[mask, "form_paste_ratio"] = np.clip(0.4 + 0.5 * _beta(rng, 3, 2, k), 0, 1)
    frame.loc[mask, "session_duration_s"] = np.clip(rng.lognormal(3.7, 0.4, k), 10.0, 600.0)


def _apply_document_forgery(frame: pd.DataFrame, rng: np.random.Generator) -> None:
    """A real person presenting a fabricated or altered document.

    Liveness and the face are genuine, so PAD and injection signals stay clean.
    The document stage fires — and the chip is missing, because a forger cannot
    mint a chip signed by the issuing country.
    """
    mask = _mask_for(frame, FraudType.DOCUMENT_FORGERY)
    k = int(mask.sum())
    if k == 0:
        return

    frame.loc[mask, "doc_tamper_score"] = np.clip(0.30 + 0.6 * _beta(rng, 3, 2, k), 0, 1)
    frame.loc[mask, "doc_template_match_score"] = np.clip(0.2 + 0.5 * _beta(rng, 2, 4, k), 0, 1)
    frame.loc[mask, "doc_mrz_checksum_valid"] = _bernoulli(rng, 0.72, k)
    # A clean forgery often reads *better* than a worn genuine document, which
    # is exactly why doc_ocr_confidence carries no monotone constraint.
    frame.loc[mask, "doc_ocr_confidence"] = np.clip(0.75 + 0.24 * _beta(rng, 5, 2, k), 0, 1)
    frame.loc[mask, "nfc_chip_read"] = _bernoulli(rng, 0.12, k)
    frame.loc[mask, "nfc_passive_auth_passed"] = np.where(
        frame.loc[mask, "nfc_chip_read"].to_numpy() > 0, _bernoulli(rng, 0.05, k), 0.0
    )
    frame.loc[mask, "face_match_score"] = np.clip(0.55 + 0.4 * _beta(rng, 4, 2, k), 0, 1)
    frame.loc[mask, "ip_country_mismatch"] = _bernoulli(rng, 0.35, k)


def _apply_synthetic_identity_rings(frame: pd.DataFrame, rng: np.random.Generator) -> None:
    """Fabricated identities farmed at scale by an organised group.

    Every individual session is clean, because every individual session *is*
    clean: a real operator, a real face, a document good enough to pass. The
    only thing that gives the ring away is that its members share devices, IPs
    and PII with each other.

    A stack of biometric and document checks alone cannot see this. That is the
    single strongest argument for the fusion model in this package.
    """
    mask = _mask_for(frame, FraudType.SYNTHETIC_ID)
    k = int(mask.sum())
    if k == 0:
        return

    ring_size = rng.integers(4, 40, size=k).astype(float)
    frame.loc[mask, "graph_ring_size"] = ring_size
    frame.loc[mask, "shared_pii_count"] = np.clip(
        rng.poisson(np.maximum(ring_size * 0.35, 1.0)), 1, None
    ).astype(float)
    frame.loc[mask, "applications_per_device_7d"] = np.clip(
        rng.poisson(np.maximum(ring_size * 0.30, 1.0)), 1, None
    ).astype(float)
    frame.loc[mask, "applications_per_ip_7d"] = np.clip(
        rng.poisson(np.maximum(ring_size * 0.55, 1.0)), 1, None
    ).astype(float)

    # Ring members genuinely share hardware, so the device_id column has to
    # reflect it. This is also what the grouped split has to defend against:
    # a ring straddling train and test would leak.
    ring_ids = rng.integers(0, max(k // 6, 1), size=k)
    frame.loc[mask, "device_id"] = [f"ringdev_{i:05d}" for i in ring_ids]

    frame.loc[mask, "ip_is_hosting"] = _bernoulli(rng, 0.25, k)
    frame.loc[mask, "form_paste_ratio"] = np.clip(0.35 + 0.5 * _beta(rng, 3, 3, k), 0, 1)
    frame.loc[mask, "session_duration_s"] = np.clip(rng.lognormal(4.3, 0.35, k), 20.0, 900.0)


#: Signals an unlucky genuine applicant can trip for innocent reasons.
_TRIPPABLE_SIGNALS: tuple[str, ...] = (
    "pad_attack_score",
    "doc_tamper_score",
    "doc_print_recapture_score",
    "frame_timing_anomaly_score",
    "virtual_camera_detected",
    "device_is_emulator",
    "ip_is_hosting",
)


def _apply_confusable_cases(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    config: GeneratorConfig,
    baseline: pd.DataFrame,
) -> None:
    """Blur the boundary, in both directions.

    Without this the classes are separable and the model scores a perfect
    PR-AUC — which is not a nice result, it is a sign the data-set is not
    posing the problem. Two consequences follow and both matter: thresholds
    tuned on separable data collapse the step-up band to nothing, and a
    calibration curve fitted to scores that are all 0 or 1 says nothing.

    Two mechanisms, both of which exist in production:

    * **Sophisticated attacks** invest in suppressing whatever gives them away
      — residential proxies, real unrooted handsets, well-made artefacts,
      documents good enough to pass a template check. Every signal the attack
      moved is interpolated back toward that row's own honest baseline, so
      sophistication covers the protective signals (attestation, NFC, template
      match) as well as the incriminating ones.
    * **Unlucky genuine applicants** trip attack signals innocently:
      screen-sharing software registering a virtual camera, a corporate VPN
      looking like hosting, a dark room raising the PAD score, a cracked
      screen reading as print recapture.
    """
    n = len(frame)
    is_fraud = frame[LABEL].to_numpy() == 1

    sophisticated = is_fraud & (rng.random(n) < config.sophisticated_attack_rate)
    if sophisticated.any():
        # retained near 1 leaves an obvious attack; near 0 leaves one only the
        # graph and behavioural stages can see. Beta(1.6, 3) leans toward the
        # latter, which is the direction the threat landscape has moved.
        retained = rng.beta(1.6, 3.0, size=int(sophisticated.sum()))[:, None]

        attacked = frame.loc[sophisticated, list(_NUMERIC_SIGNALS)].to_numpy(dtype="float64")
        honest = baseline.loc[sophisticated].to_numpy(dtype="float64")

        blended = honest + retained * (attacked - honest)
        frame.loc[sophisticated, list(_NUMERIC_SIGNALS)] = blended

    unlucky = ~is_fraud & (rng.random(n) < config.unlucky_genuine_rate)
    if unlucky.any():
        positions = np.flatnonzero(unlucky)
        # One or two signals, not all of them: a genuine applicant who trips
        # every check at once is, realistically, not genuine.
        for position in positions:
            chosen = rng.choice(_TRIPPABLE_SIGNALS, size=int(rng.integers(1, 3)), replace=False)
            for column in chosen:
                frame.iloc[position, frame.columns.get_loc(column)] = float(
                    np.clip(0.35 + 0.6 * rng.beta(3, 3), 0.0, 1.0)
                )
        frame.loc[unlucky, "retry_count"] = rng.poisson(1.5, int(unlucky.sum())).astype(float)


def _apply_missingness(
    frame: pd.DataFrame, rng: np.random.Generator, config: GeneratorConfig
) -> None:
    """Blank out the optional signals the way real capture does.

    NFC is the important one: it is absent for most sessions because of the
    *handset*, not the applicant, so its absence is close to missing-at-random
    and must not by itself imply fraud. The forgery typology above already
    encodes the informative side of it.
    """
    n = len(frame)

    no_chip_hardware = rng.random(n) < config.nfc_unavailable_rate
    frame.loc[no_chip_hardware, ["nfc_chip_read", "nfc_passive_auth_passed"]] = np.nan

    # A chip that was not read cannot have been authenticated.
    not_read = frame["nfc_chip_read"] == 0
    frame.loc[not_read, "nfc_passive_auth_passed"] = np.nan

    too_few_frames = rng.random(n) < config.rppg_unavailable_rate
    frame.loc[too_few_frames, "rppg_pulse_detected"] = np.nan


def split_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience: drop identifiers and slices, returning ``(signals, label)``."""
    from idv_risk.schema import SLICE_COLUMNS

    drop = [*IDENTIFIER_COLUMNS, *SLICE_COLUMNS, LABEL]
    signals = frame.drop(columns=[c for c in drop if c in frame.columns])
    return signals, frame[LABEL]


def unknown_category_frame(frame: pd.DataFrame, column: str = "device_os") -> pd.DataFrame:
    """Return a copy whose ``column`` holds a level the model never saw.

    Used by the tests that pin down the unseen-category behaviour, which is a
    real outage mode: XGBoost raises on an unknown categorical level, so a new
    device platform would take scoring down without the folding that
    :mod:`idv_risk.features` does.
    """
    out = frame.copy()
    out[column] = UNKNOWN_CATEGORY.replace("__", "") + "_os"
    return out
