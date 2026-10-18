"""The XGBoost model, its monotone constraints, and calibration.

``test_monotone_constraint_holds_end_to_end`` is the one to keep: it probes the
fitted model by sweeping a constrained signal and checking the risk never falls.
That is the property an auditor is told about and the one an attacker would
otherwise exploit, and it is checked against the real fitted booster rather
than against the parameter dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from idv_risk.calibration import (
    choose_method,
    expected_calibration_error,
    fit_calibrator,
    reliability_table,
)
from idv_risk.config import CalibrationConfig, ModelConfig
from idv_risk.features import build_features
from idv_risk.model import (
    build_classifier,
    compute_scale_pos_weight,
    feature_importance,
    predict_risk,
)
from idv_risk.schema import FEATURE_NAMES, SIGNALS_BY_NAME, Direction


class TestScalePosWeight:
    def test_is_the_negative_to_positive_ratio(self):
        labels = np.array([0] * 96 + [1] * 4)

        assert compute_scale_pos_weight(labels) == pytest.approx(24.0)

    def test_balanced_labels_give_one(self):
        assert compute_scale_pos_weight(np.array([0, 1, 0, 1])) == pytest.approx(1.0)

    def test_no_positives_does_not_divide_by_zero(self):
        assert compute_scale_pos_weight(np.zeros(10)) == 1.0


class TestBuildClassifier:
    def test_applies_the_schema_constraints(self):
        classifier = build_classifier(ModelConfig())

        constraints = classifier.get_params()["monotone_constraints"]
        assert constraints["pad_attack_score"] == 1
        assert constraints["nfc_passive_auth_passed"] == -1
        assert constraints["face_match_score"] == 0

    def test_constraints_can_be_disabled_for_ablation(self):
        classifier = build_classifier(ModelConfig(use_monotone_constraints=False))

        assert classifier.get_params().get("monotone_constraints") is None

    def test_categorical_support_is_enabled(self):
        assert build_classifier(ModelConfig()).get_params()["enable_categorical"] is True

    def test_uses_the_precision_recall_metric_for_early_stopping(self):
        """Under heavy imbalance AUC-ROC barely moves; AUC-PR does."""
        assert build_classifier(ModelConfig()).get_params()["eval_metric"] == "aucpr"

    def test_explicit_scale_pos_weight_overrides_the_computed_one(self):
        classifier = build_classifier(ModelConfig(scale_pos_weight=3.0), scale_pos_weight=99.0)

        assert classifier.get_params()["scale_pos_weight"] == 3.0


class TestFittedModel:
    def test_early_stopping_selected_fewer_trees_than_the_cap(self, trained, fast_config):
        classifier = trained.model.classifier

        assert classifier.best_iteration is not None
        assert classifier.best_iteration < fast_config.model.n_estimators

    def test_predictions_use_the_best_iteration(self, trained, frame):
        """XGBoost truncates to best_iteration by default when early stopping ran.

        Version-dependent behaviour that silently changes every score if it
        stops holding, so it is verified rather than assumed.
        """
        classifier = trained.model.classifier
        features = build_features(frame.head(200))

        default = classifier.predict_proba(features)[:, 1]
        best = classifier.predict_proba(
            features, iteration_range=(0, classifier.best_iteration + 1)
        )[:, 1]
        every_tree = classifier.predict_proba(
            features, iteration_range=(0, classifier.get_booster().num_boosted_rounds())
        )[:, 1]

        np.testing.assert_allclose(default, best)
        assert not np.allclose(default, every_tree)

    def test_probabilities_are_in_range(self, trained, frame):
        risk = predict_risk(trained.model.classifier, build_features(frame.head(300)))

        assert risk.min() >= 0.0
        assert risk.max() <= 1.0

    def test_ranks_fraud_above_genuine(self, trained, frame):
        """Stated as a ranking property rather than a ratio of means: raw
        scores are inflated by scale_pos_weight, so their absolute size is not
        meaningful until calibration."""
        from sklearn.metrics import roc_auc_score

        sample = frame.tail(1500)
        risk = predict_risk(trained.model.classifier, build_features(sample))
        is_fraud = sample["is_fraud"].to_numpy()

        assert risk[is_fraud == 1].mean() > risk[is_fraud == 0].mean()
        assert roc_auc_score(is_fraud, risk) > 0.9

    @pytest.mark.parametrize(
        "name",
        [n for n in FEATURE_NAMES if SIGNALS_BY_NAME[n].direction is not Direction.UNCONSTRAINED],
    )
    def test_monotone_constraint_holds_end_to_end(self, trained, frame, name):
        """Sweep one constrained signal and check risk moves the declared way.

        Probed on the fitted booster, from a real median application, so this
        catches a constraint that was declared but not actually applied.
        """
        signal = SIGNALS_BY_NAME[name]
        features = build_features(frame.head(400))

        probe = pd.concat([features.iloc[[0]]] * 25, ignore_index=True)
        for column in probe.columns:
            if isinstance(probe[column].dtype, pd.CategoricalDtype):
                continue
            probe[column] = features[column].median()

        low, high = (signal.minimum or 0.0), (signal.maximum or features[name].max())
        probe[name] = np.linspace(low, high, len(probe))

        risk = predict_risk(trained.model.classifier, probe)

        if signal.direction is Direction.INCREASES_RISK:
            assert np.all(np.diff(risk) >= -1e-9), f"{name} should not lower risk as it rises"
        else:
            assert np.all(np.diff(risk) <= 1e-9), f"{name} should not raise risk as it rises"

    def test_handles_missing_optional_signals_without_imputation(self, trained, frame):
        """The reason a tree ensemble suits this problem."""
        sample = frame.head(200).copy()
        sample["nfc_chip_read"] = np.nan
        sample["nfc_passive_auth_passed"] = np.nan

        risk = predict_risk(trained.model.classifier, build_features(sample))

        assert np.isfinite(risk).all()


class TestFeatureImportance:
    def test_returns_a_value_for_every_feature(self, trained):
        importance = feature_importance(trained.model.classifier)

        assert set(importance.index) == set(FEATURE_NAMES)

    def test_is_sorted_descending(self, trained):
        importance = feature_importance(trained.model.classifier)

        assert list(importance) == sorted(importance, reverse=True)

    def test_injection_and_document_signals_carry_weight(self, trained):
        """Sanity check that the model is using more than one signal family."""
        importance = feature_importance(trained.model.classifier)
        top = set(importance.head(12).index)

        assert top & {
            "virtual_camera_detected",
            "device_attestation_passed",
            "frame_timing_anomaly_score",
        }
        assert top & {"doc_tamper_score", "doc_template_match_score", "nfc_passive_auth_passed"}


class TestChooseMethod:
    def test_prefers_isotonic_with_enough_data(self):
        labels = np.array([1] * 100 + [0] * 5000)

        assert choose_method(labels, CalibrationConfig()) == "isotonic"

    def test_falls_back_on_too_few_rows(self):
        labels = np.array([1] * 60 + [0] * 100)

        assert choose_method(labels, CalibrationConfig()) == "sigmoid"

    def test_falls_back_on_too_few_positives(self):
        labels = np.array([1] * 10 + [0] * 5000)

        assert choose_method(labels, CalibrationConfig()) == "sigmoid"


class TestCalibration:
    def test_wraps_without_refitting_the_classifier(self, trained, frame):
        """FrozenEstimator replaced cv='prefit', deprecated in scikit-learn 1.6.

        Without it the classifier would be cloned and retrained on the
        calibration split, discarding the early-stopped fit.
        """
        classifier = trained.model.classifier
        before = classifier.best_iteration

        sample = frame.head(1500)
        fit_calibrator(classifier, build_features(sample), sample["is_fraud"])

        assert classifier.best_iteration == before

    def test_produces_a_calibrated_classifier(self, model):
        assert isinstance(model.calibrator, CalibratedClassifierCV)

    def test_underlying_estimator_is_the_fitted_booster(self, model):
        assert isinstance(model.classifier, xgb.XGBClassifier)

    def test_calibration_improves_the_expected_calibration_error(self, trained, frame):
        """The reason calibration is a dependency, not a nicety: the decision
        bands are absolute probabilities."""
        _, _, test_frame = trained.split.frames(frame)
        features = build_features(test_frame)
        labels = test_frame["is_fraud"].to_numpy()

        raw = trained.model.classifier.predict_proba(features)[:, 1]
        calibrated = trained.model.calibrator.predict_proba(features)[:, 1]

        assert expected_calibration_error(calibrated, labels) < expected_calibration_error(
            raw, labels
        )

    def test_calibration_preserves_ranking(self, trained, frame):
        """Both methods are monotone, so PR-AUC must not move."""
        from sklearn.metrics import average_precision_score

        _, _, test_frame = trained.split.frames(frame)
        features = build_features(test_frame)
        labels = test_frame["is_fraud"].to_numpy()

        raw = trained.model.classifier.predict_proba(features)[:, 1]
        calibrated = trained.model.calibrator.predict_proba(features)[:, 1]

        assert average_precision_score(labels, calibrated) == pytest.approx(
            average_precision_score(labels, raw), abs=0.05
        )


class TestExpectedCalibrationError:
    def test_perfect_calibration_scores_zero(self):
        probabilities = np.concatenate([np.zeros(500), np.ones(500)])
        labels = np.concatenate([np.zeros(500), np.ones(500)])

        assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)

    def test_confidently_wrong_scores_one(self):
        probabilities = np.ones(100)
        labels = np.zeros(100)

        assert expected_calibration_error(probabilities, labels) == pytest.approx(1.0)

    def test_a_model_predicting_the_base_rate_is_well_calibrated(self):
        """And useless — which is why ECE is never reported without PR-AUC."""
        labels = np.array([1] * 100 + [0] * 900)
        probabilities = np.full(1000, 0.1)

        assert expected_calibration_error(probabilities, labels) < 0.01

    def test_probability_of_one_lands_in_the_last_bin(self):
        """np.digitize would otherwise put it one bin past the end."""
        expected_calibration_error(np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="zero predictions"):
            expected_calibration_error(np.array([]), np.array([]))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="disagree on length"):
            expected_calibration_error(np.zeros(3), np.zeros(2))


class TestReliabilityTable:
    def test_has_one_row_per_bin(self):
        table = reliability_table(
            np.linspace(0, 1, 100), np.random.default_rng(0).integers(0, 2, 100)
        )

        assert len(table) == 10

    def test_counts_sum_to_the_population(self):
        table = reliability_table(np.linspace(0, 1, 100), np.zeros(100))

        assert table["count"].sum() == 100

    def test_empty_bins_report_nan_rather_than_zero(self):
        table = reliability_table(np.zeros(50), np.zeros(50))

        assert table.loc[table["count"] == 0, "observed_rate"].isna().all()
