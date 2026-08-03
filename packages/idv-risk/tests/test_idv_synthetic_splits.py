"""The generated data-set and the leakage-free split.

``TestTypologyBlindSpots`` is the load-bearing part: it pins down that each
fraud typology is invisible to a different signal family, which is the premise
the whole fusion model rests on. If those assertions stop holding, the
generator has stopped posing the problem and every accuracy number in the
package becomes decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idv_risk.schema import FEATURE_NAMES, LABEL, SIGNALS_BY_NAME
from idv_risk.splits import (
    assert_chronological,
    assert_no_group_overlap,
    purge_seen_groups,
    time_grouped_split,
    time_split,
)
from idv_risk.synthetic import ATTACK_TYPES, FraudType, GeneratorConfig, generate


class TestGenerate:
    def test_produces_the_requested_number_of_rows(self):
        assert len(generate(GeneratorConfig(n_applications=500, seed=1))) == 500

    def test_is_deterministic_for_a_given_seed(self):
        a = generate(GeneratorConfig(n_applications=400, seed=3))
        b = generate(GeneratorConfig(n_applications=400, seed=3))

        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_give_different_data(self):
        a = generate(GeneratorConfig(n_applications=400, seed=3))
        b = generate(GeneratorConfig(n_applications=400, seed=4))

        assert not a["pad_attack_score"].equals(b["pad_attack_score"])

    def test_carries_every_model_signal(self, frame):
        assert set(FEATURE_NAMES) <= set(frame.columns)

    def test_fraud_rate_is_near_the_requested_rate(self):
        generated = generate(GeneratorConfig(n_applications=8000, fraud_rate=0.04, seed=5))

        assert generated[LABEL].mean() == pytest.approx(0.04, abs=0.01)

    def test_label_agrees_with_the_typology_column(self, frame):
        expected = (frame["fraud_type"] != FraudType.GENUINE.value).astype(int)

        pd.testing.assert_series_equal(frame[LABEL], expected, check_names=False)

    def test_every_typology_is_represented(self, frame):
        present = set(frame["fraud_type"].unique())

        assert present == {FraudType.GENUINE.value} | {t.value for t in ATTACK_TYPES}

    def test_timestamps_are_in_order(self, frame):
        assert frame["timestamp"].is_monotonic_increasing

    def test_optional_signals_carry_missing_values(self, frame):
        for name in ("nfc_chip_read", "rppg_pulse_detected", "pad_active_challenge_passed"):
            assert frame[name].isna().any(), f"{name} should sometimes be absent"

    def test_required_signals_are_never_missing(self, frame):
        for name in FEATURE_NAMES:
            if not SIGNALS_BY_NAME[name].optional:
                assert not frame[name].isna().any(), name

    def test_unread_chip_cannot_have_been_authenticated(self, frame):
        not_read = frame["nfc_chip_read"] == 0

        assert frame.loc[not_read, "nfc_passive_auth_passed"].isna().all()

    def test_unit_range_signals_stay_in_range(self, frame):
        for name in FEATURE_NAMES:
            signal = SIGNALS_BY_NAME[name]
            if signal.is_categorical or signal.maximum is None:
                continue
            values = frame[name].dropna()
            assert values.min() >= signal.minimum, name
            assert values.max() <= signal.maximum, name

    @pytest.mark.parametrize("bad", [{"fraud_rate": 0.0}, {"fraud_rate": 1.0}])
    def test_rejects_a_degenerate_fraud_rate(self, bad):
        with pytest.raises(ValueError, match="fraud_rate"):
            generate(GeneratorConfig(**bad))

    def test_rejects_an_empty_population(self):
        with pytest.raises(ValueError, match="n_applications"):
            generate(GeneratorConfig(n_applications=0))

    def test_rejects_a_typology_mix_that_does_not_sum_to_one(self):
        config = GeneratorConfig(typology_mix={FraudType.PRESENTATION: 0.5})

        with pytest.raises(ValueError, match="must sum to 1"):
            generate(config)


class TestTypologyBlindSpots:
    """Each attack must be invisible to a different part of the stack."""

    @pytest.fixture(scope="class")
    def by_type(self):
        generated = generate(GeneratorConfig(n_applications=20_000, seed=9))
        return {name: group for name, group in generated.groupby("fraud_type")}

    def test_presentation_attacks_raise_the_pad_score(self, by_type):
        assert (
            by_type["presentation"]["pad_attack_score"].mean()
            > by_type["genuine"]["pad_attack_score"].mean() * 2
        )

    def test_injection_attacks_do_not_raise_the_pad_score(self, by_type):
        """The whole point: nothing was presented to a camera, so PAD is blind."""
        assert by_type["injection"]["pad_attack_score"].mean() == pytest.approx(
            by_type["genuine"]["pad_attack_score"].mean(), abs=0.06
        )

    def test_injection_attacks_match_the_face_better_than_genuine_users(self, by_type):
        """Synthesised media is cleaner than a real selfie in a hallway."""
        assert (
            by_type["injection"]["face_match_score"].mean()
            > by_type["genuine"]["face_match_score"].mean()
        )

    def test_injection_attacks_fail_device_attestation(self, by_type):
        assert (
            by_type["injection"]["device_attestation_passed"].mean()
            < by_type["genuine"]["device_attestation_passed"].mean() * 0.8
        )

    def test_document_forgery_leaves_liveness_clean(self, by_type):
        """A real person is holding the fake document."""
        assert by_type["document_forgery"]["pad_attack_score"].mean() == pytest.approx(
            by_type["genuine"]["pad_attack_score"].mean(), abs=0.06
        )

    def test_document_forgery_raises_the_tamper_score(self, by_type):
        assert (
            by_type["document_forgery"]["doc_tamper_score"].mean()
            > by_type["genuine"]["doc_tamper_score"].mean() * 2
        )

    def test_document_forgery_rarely_produces_a_chip(self, by_type):
        """A forger cannot mint a chip signed by the issuing country."""
        assert by_type["document_forgery"]["nfc_chip_read"].mean(skipna=True) < 0.5

    def test_synthetic_identity_leaves_every_session_signal_clean(self, by_type):
        """The one a pure biometrics stack cannot see at all."""
        rings = by_type["synthetic_id"]
        genuine = by_type["genuine"]

        for name in ("pad_attack_score", "doc_tamper_score", "frame_timing_anomaly_score"):
            assert rings[name].mean() == pytest.approx(genuine[name].mean(), abs=0.06), name

    def test_synthetic_identity_shows_up_only_in_the_graph(self, by_type):
        rings = by_type["synthetic_id"]
        genuine = by_type["genuine"]

        assert rings["graph_ring_size"].mean() > genuine["graph_ring_size"].mean() * 3
        assert rings["shared_pii_count"].mean() > genuine["shared_pii_count"].mean() * 3

    def test_sophisticated_attacks_create_class_overlap(self, by_type):
        """Some attacks must look clean, or the problem is not a problem.

        Separable classes give a perfect PR-AUC, collapse the step-up band and
        make the calibration curve meaningless.
        """
        attacks = pd.concat([by_type[t.value] for t in ATTACK_TYPES])
        genuine_p95 = by_type["genuine"]["pad_attack_score"].quantile(0.95)

        below = (attacks["pad_attack_score"] <= genuine_p95).mean()
        assert 0.2 < below < 0.95, f"{below:.2%} of attacks look benign to PAD"

    def test_some_genuine_applicants_trip_an_attack_signal(self, by_type):
        genuine = by_type["genuine"]

        assert (genuine["pad_attack_score"] > 0.35).mean() > 0.005


class TestTimeSplit:
    def test_periods_are_chronological(self, frame):
        split = time_split(frame)

        assert_chronological(frame, split)

    def test_periods_cover_every_row_exactly_once(self, frame):
        split = time_split(frame)

        combined = np.concatenate([split.train, split.valid, split.test])
        assert sorted(combined) == sorted(frame.index)

    def test_fractions_are_respected(self, frame):
        split = time_split(frame, valid_fraction=0.2, test_fraction=0.1)

        n_train, n_valid, n_test = split.sizes
        assert n_valid == pytest.approx(len(frame) * 0.2, rel=0.02)
        assert n_test == pytest.approx(len(frame) * 0.1, rel=0.02)

    def test_rejects_fractions_leaving_no_training_data(self, frame):
        with pytest.raises(ValueError, match="must be below 1.0"):
            time_split(frame, valid_fraction=0.6, test_fraction=0.6)

    def test_rejects_a_missing_time_column(self, frame):
        with pytest.raises(KeyError, match="timestamp"):
            time_split(frame.drop(columns=["timestamp"]))


class TestPurging:
    def test_repeat_applicants_are_removed_from_later_periods(self, frame):
        split = purge_seen_groups(frame, time_split(frame))

        assert_no_group_overlap(frame, split)

    def test_purging_only_shrinks_the_later_periods(self, frame):
        raw = time_split(frame)
        purged = purge_seen_groups(frame, raw)

        assert len(purged.train) == len(raw.train)
        assert len(purged.valid) <= len(raw.valid)
        assert len(purged.test) <= len(raw.test)

    def test_purging_removes_only_a_small_share(self, frame):
        """A purge that eats the evaluation set is a broken purge.

        Uniformly-scattered repeat applicants would remove most of it; the
        generator makes repeats temporally local, as real retries are.
        """
        raw = time_split(frame)
        purged = purge_seen_groups(frame, raw)

        assert len(purged.test) > 0.8 * len(raw.test)

    def test_devices_are_deliberately_not_purged(self, frame):
        """Sharing a device with a flagged applicant is signal, not leakage.

        Purging it would delete the synthetic-identity rings from evaluation
        and measure the graph stage on a world where it cannot work.
        """
        split = time_grouped_split(frame)

        train_devices = set(frame.loc[split.train, "device_id"])
        test_devices = set(frame.loc[split.test, "device_id"])
        assert train_devices & test_devices

    def test_every_typology_survives_into_the_test_period(self, frame):
        """The regression this guards is real: purging on device_id silently
        deleted the whole synthetic_id typology from the test split."""
        split = time_grouped_split(frame)

        present = set(frame.loc[split.test, "fraud_type"].unique())
        assert {t.value for t in ATTACK_TYPES} <= present

    def test_purging_on_a_missing_column_raises(self, frame):
        with pytest.raises(KeyError, match="nonexistent"):
            purge_seen_groups(frame, time_split(frame), ("nonexistent",))

    def test_no_purge_columns_is_a_no_op(self, frame):
        raw = time_split(frame)

        assert purge_seen_groups(frame, raw, ()).sizes == raw.sizes


class TestSplitHelpers:
    def test_frames_returns_the_three_periods(self, frame):
        split = time_grouped_split(frame)
        train, valid, test = split.frames(frame)

        assert (len(train), len(valid), len(test)) == split.sizes

    def test_split_unpacks_as_a_triple(self, frame):
        train, valid, test = time_grouped_split(frame)

        assert len(train) + len(valid) + len(test) <= len(frame)

    def test_overlap_assertion_catches_a_deliberately_broken_split(self, frame):
        from idv_risk.splits import Split

        broken = Split(
            train=frame.index[:100], valid=frame.index[50:150], test=frame.index[150:200]
        )

        with pytest.raises(AssertionError, match="shared between"):
            assert_no_group_overlap(frame, broken)

    def test_chronology_assertion_catches_a_shuffled_split(self, frame):
        from idv_risk.splits import Split

        broken = Split(train=frame.index[-100:], valid=frame.index[:100], test=frame.index[100:200])

        with pytest.raises(AssertionError, match="extends past"):
            assert_chronological(frame, broken)
