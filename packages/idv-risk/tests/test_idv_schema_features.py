"""The signal contract and the train/serve feature builder.

The unseen-category test is the one that matters most here: XGBoost raises on a
categorical level it did not see in training, so without the folding in
``features.py`` a new device platform would take scoring down rather than
degrade it. That behaviour was verified against the installed XGBoost before
this package was designed around it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idv_risk.features import MissingSignalError, build_features, validate_features
from idv_risk.schema import (
    FEATURE_NAMES,
    SIGNALS,
    SIGNALS_BY_NAME,
    UNKNOWN_CATEGORY,
    Direction,
    Stage,
    categorical_names,
    feature_names_for_stages,
    monotone_constraints,
    optional_names,
)


class TestSchema:
    def test_feature_names_are_unique(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_every_signal_has_a_description(self):
        for signal in SIGNALS:
            assert signal.description.strip(), f"{signal.name} has no description"

    def test_every_stage_contributes_at_least_one_signal(self):
        covered = {signal.stage for signal in SIGNALS}
        assert covered == set(Stage)

    def test_monotone_constraints_cover_every_feature(self):
        assert set(monotone_constraints()) == set(FEATURE_NAMES)

    def test_constraint_encoding_matches_xgboost(self):
        assert Direction.INCREASES_RISK.xgboost_constraint == 1
        assert Direction.DECREASES_RISK.xgboost_constraint == -1
        assert Direction.UNCONSTRAINED.xgboost_constraint == 0

    def test_attack_scores_push_risk_up(self):
        for name in ("pad_attack_score", "doc_tamper_score", "virtual_camera_detected"):
            assert SIGNALS_BY_NAME[name].direction is Direction.INCREASES_RISK

    def test_cryptographic_proofs_push_risk_down(self):
        for name in ("nfc_passive_auth_passed", "device_attestation_passed"):
            assert SIGNALS_BY_NAME[name].direction is Direction.DECREASES_RISK

    def test_attacker_controllable_signals_are_left_unconstrained(self):
        """A monotone constraint here would hand the attacker a direction to push.

        An injection attack optimises the face match upwards, and a clean
        forgery reads better than a worn genuine document. Constraining either
        would encode the careless-attacker case and be wrong for the careful one.
        """
        assert SIGNALS_BY_NAME["face_match_score"].direction is Direction.UNCONSTRAINED
        assert SIGNALS_BY_NAME["doc_ocr_confidence"].direction is Direction.UNCONSTRAINED

    def test_only_nfc_and_capture_dependent_signals_are_optional(self):
        assert set(optional_names()) == {
            "nfc_chip_read",
            "nfc_passive_auth_passed",
            "pad_active_challenge_passed",
            "rppg_pulse_detected",
        }

    def test_categorical_levels_include_the_unknown_sentinel(self):
        for name in categorical_names():
            assert UNKNOWN_CATEGORY in (SIGNALS_BY_NAME[name].categories or ())

    def test_stage_selection_preserves_model_order(self):
        selected = feature_names_for_stages(Stage.GRAPH, Stage.DOCUMENT)
        assert list(selected) == [n for n in FEATURE_NAMES if n in set(selected)]

    def test_stage_selection_is_a_strict_subset(self):
        graph_only = feature_names_for_stages(Stage.GRAPH)
        assert set(graph_only) < set(FEATURE_NAMES)
        assert all(SIGNALS_BY_NAME[n].stage is Stage.GRAPH for n in graph_only)


class TestBuildFeatures:
    def test_column_order_matches_the_contract(self, small_frame):
        features = build_features(small_frame)
        assert tuple(features.columns) == FEATURE_NAMES

    def test_column_order_is_independent_of_input_order(self, small_frame):
        shuffled = small_frame[list(reversed(small_frame.columns))]

        features = build_features(shuffled)

        assert tuple(features.columns) == FEATURE_NAMES

    def test_row_count_and_index_are_preserved(self, small_frame):
        features = build_features(small_frame)

        assert len(features) == len(small_frame)
        pd.testing.assert_index_equal(features.index, small_frame.index)

    def test_extra_columns_are_ignored(self, small_frame):
        extra = small_frame.assign(some_warehouse_column=1)

        assert tuple(build_features(extra).columns) == FEATURE_NAMES

    def test_missing_required_signal_is_named_in_the_error(self, small_frame):
        without = small_frame.drop(columns=["pad_attack_score"])

        with pytest.raises(MissingSignalError, match="pad_attack_score"):
            build_features(without)

    def test_missing_optional_signal_becomes_an_all_nan_column(self, small_frame):
        """A capability that is not deployed yet is a legitimate state."""
        without = small_frame.drop(columns=["nfc_chip_read"])

        features = build_features(without)

        assert features["nfc_chip_read"].isna().all()

    def test_optional_signals_keep_their_nans_rather_than_being_imputed(self, small_frame):
        features = build_features(small_frame)

        # XGBoost routes NaN itself; imputing would destroy the information
        # that no chip was read.
        assert features["nfc_chip_read"].isna().any()

    def test_required_signals_have_no_nans(self, small_frame):
        features = build_features(small_frame)

        for name in FEATURE_NAMES:
            if not SIGNALS_BY_NAME[name].optional and not SIGNALS_BY_NAME[name].is_categorical:
                assert not features[name].isna().any(), name

    def test_out_of_range_values_are_clipped_not_rejected(self, small_frame):
        """A vendor shipping 1.02 should not take the request down."""
        broken = small_frame.copy()
        broken.loc[broken.index[0], "pad_attack_score"] = 1.4
        broken.loc[broken.index[1], "pad_attack_score"] = -0.3

        features = build_features(broken)

        assert features["pad_attack_score"].max() <= 1.0
        assert features["pad_attack_score"].min() >= 0.0

    def test_non_numeric_junk_becomes_nan_rather_than_raising(self, small_frame):
        broken = small_frame.copy()
        broken["pad_attack_score"] = broken["pad_attack_score"].astype(object)
        broken.loc[broken.index[0], "pad_attack_score"] = "not a number"

        features = build_features(broken)

        assert np.isnan(features["pad_attack_score"].iloc[0])

    def test_subset_of_features_can_be_built(self, small_frame):
        subset = feature_names_for_stages(Stage.GRAPH)

        features = build_features(small_frame, subset)

        assert tuple(features.columns) == subset


class TestCategoricalHandling:
    def test_known_levels_are_preserved(self, small_frame):
        features = build_features(small_frame)

        assert set(features["device_os"].dropna().unique()) <= {"android", "ios", "web"}

    def test_unseen_level_folds_to_the_unknown_sentinel(self, small_frame):
        """Without this, `predict` raises on a device platform that did not
        exist at training time — an outage, not a degradation."""
        modified = small_frame.copy()
        modified["device_os"] = "harmonyos"

        features = build_features(modified)

        assert (features["device_os"] == UNKNOWN_CATEGORY).all()

    def test_absent_value_stays_absent_rather_than_becoming_unknown(self, small_frame):
        """ "Not collected" and "collected but unrecognised" are different facts."""
        modified = small_frame.copy()
        modified.loc[modified.index[0], "device_os"] = None

        features = build_features(modified)

        assert pd.isna(features["device_os"].iloc[0])

    def test_dtype_carries_the_full_declared_level_set(self, small_frame):
        features = build_features(small_frame)

        assert set(features["device_os"].dtype.categories) == set(
            SIGNALS_BY_NAME["device_os"].categories
        )

    def test_level_set_is_stable_across_calls_with_different_data(self, small_frame):
        """XGBoost encodes categories by position, so the level set must not
        depend on which rows happen to be in the batch."""
        one_platform = small_frame.copy()
        one_platform["device_os"] = "ios"

        a = build_features(small_frame)["device_os"].dtype.categories
        b = build_features(one_platform)["device_os"].dtype.categories

        assert list(a) == list(b)


class TestValidateFeatures:
    def test_accepts_a_well_built_frame(self, small_frame):
        validate_features(build_features(small_frame))

    def test_rejects_reordered_columns(self, small_frame):
        features = build_features(small_frame)
        reordered = features[list(reversed(FEATURE_NAMES))]

        with pytest.raises(ValueError, match="feature order mismatch"):
            validate_features(reordered)

    def test_rejects_a_missing_required_value(self, small_frame):
        features = build_features(small_frame)
        features.loc[features.index[0], "pad_attack_score"] = np.nan

        with pytest.raises(ValueError, match="missing values"):
            validate_features(features)

    def test_rejects_a_stringified_numeric_column(self, small_frame):
        features = build_features(small_frame)
        features["pad_attack_score"] = features["pad_attack_score"].astype(str)

        with pytest.raises(ValueError, match="expected float dtype"):
            validate_features(features)

    def test_rejects_a_categorical_carrying_an_undeclared_level(self, small_frame):
        features = build_features(small_frame)
        features["device_os"] = pd.Categorical(
            ["android"] * len(features), categories=["android", "plan9"]
        )

        with pytest.raises(ValueError, match="unexpected categories"):
            validate_features(features)
