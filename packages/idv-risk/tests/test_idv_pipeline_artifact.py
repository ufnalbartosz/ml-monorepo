"""The end-to-end pipeline, the deployable bundle, and the ablations.

``TestAblations`` is the argument for the whole package: it trains models on
individual signal families and shows each one has a blind spot that fusion
closes. If a biometrics-only model could catch the synthetic-identity rings,
this package would not need to exist.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from idv_risk.artifact import RiskModel, dataframe_fingerprint
from idv_risk.cli import main as cli_main
from idv_risk.config import ModelConfig, TrainingConfig
from idv_risk.decision import Decision
from idv_risk.pipeline import evaluate, summarise, train
from idv_risk.schema import LABEL, Stage, feature_names_for_stages
from idv_risk.synthetic import GeneratorConfig, generate


class TestTrain:
    def test_produces_a_usable_model(self, trained):
        assert trained.model.classifier is not None
        assert trained.model.calibrator is not None

    def test_reports_on_both_validation_and_test(self, trained):
        assert "overall" in trained.validation_report
        assert "overall" in trained.test_report

    def test_learns_something_useful(self, trained):
        """A PR-AUC near the base rate would mean the model learned nothing."""
        overall = trained.test_report["overall"]

        assert overall["pr_auc"] > 0.6
        assert overall["pr_auc"] > overall["base_rate"] * 5

    def test_is_calibrated_enough_for_the_thresholds_to_mean_something(self, trained):
        assert trained.test_report["overall"]["ece"] < 0.05

    def test_catches_every_typology_at_least_sometimes(self, trained):
        by_type = {
            row["group"]: row
            for row in trained.test_report["recall_by_fraud_type"]
            if row["n_positive"] > 0
        }

        assert len(by_type) == 4, f"expected four typologies, got {sorted(by_type)}"
        for name, row in by_type.items():
            assert row["recall"] > 0.4, f"{name} recall {row['recall']:.2f}"

    def test_thresholds_are_ordered(self, trained):
        thresholds = trained.model.thresholds

        assert thresholds.accept_below <= thresholds.reject_at_or_above

    def test_splits_do_not_leak_applicants(self, trained, frame):
        from idv_risk.splits import assert_no_group_overlap

        assert_no_group_overlap(frame, trained.split)

    def test_reports_the_cost_baselines_for_comparison(self, trained):
        assert set(trained.test_report["baseline_costs"]) == {
            "accept_all",
            "reject_all",
            "step_up_all",
        }

    def test_records_feature_importance(self, trained):
        assert trained.test_report["feature_importance"]

    def test_records_a_reliability_table(self, trained):
        assert len(trained.test_report["reliability"]) == 10

    def test_missing_label_column_is_named_in_the_error(self, small_frame):
        with pytest.raises(KeyError, match=LABEL):
            train(small_frame.drop(columns=[LABEL]))


class TestProvenance:
    def test_records_the_training_shape(self, trained):
        provenance = trained.model.provenance

        assert provenance.training_rows > 0
        assert 0.0 < provenance.training_fraud_rate < 1.0

    def test_records_library_versions(self, trained):
        assert {"xgboost", "scikit-learn", "pandas", "numpy"} <= set(
            trained.model.provenance.library_versions
        )

    def test_records_the_metrics_the_model_shipped_with(self, trained):
        assert "test" in trained.model.provenance.metrics

    def test_fingerprint_is_stable_for_identical_data(self, small_frame):
        assert dataframe_fingerprint(small_frame) == dataframe_fingerprint(small_frame.copy())

    def test_fingerprint_ignores_column_order(self, small_frame):
        reordered = small_frame[list(reversed(small_frame.columns))]

        assert dataframe_fingerprint(reordered) == dataframe_fingerprint(small_frame)

    def test_fingerprint_changes_when_a_value_changes(self, small_frame):
        modified = small_frame.copy()
        modified.loc[modified.index[0], "pad_attack_score"] = 0.12345

        assert dataframe_fingerprint(modified) != dataframe_fingerprint(small_frame)


class TestArtifactRoundTrip:
    def test_saves_every_component(self, model, tmp_path):
        model.save(tmp_path / "bundle")

        for name in ("booster.json", "calibrator.joblib", "metadata.json"):
            assert (tmp_path / "bundle" / name).exists()

    def test_reloaded_model_scores_identically(self, model, frame, tmp_path):
        """The export contract. A bundle that scores differently from the model
        that wrote it is the worst kind of bug: silent, and only visible in
        production."""
        sample = frame.tail(500)
        before = model.risk_score(sample)

        model.save(tmp_path / "bundle")
        reloaded = RiskModel.load(tmp_path / "bundle")

        np.testing.assert_allclose(reloaded.risk_score(sample), before, rtol=1e-9)

    def test_reloaded_model_decides_identically(self, model, frame, tmp_path):
        sample = frame.tail(500)
        before = model.decide(sample)

        model.save(tmp_path / "bundle")
        reloaded = RiskModel.load(tmp_path / "bundle")

        assert list(reloaded.decide(sample)) == list(before)

    def test_thresholds_survive_the_round_trip(self, model, tmp_path):
        model.save(tmp_path / "bundle")

        assert RiskModel.load(tmp_path / "bundle").thresholds == model.thresholds

    def test_feature_order_survives_the_round_trip(self, model, tmp_path):
        model.save(tmp_path / "bundle")

        assert RiskModel.load(tmp_path / "bundle").feature_names == model.feature_names

    def test_config_survives_the_round_trip(self, model, tmp_path):
        model.save(tmp_path / "bundle")

        assert RiskModel.load(tmp_path / "bundle").config == model.config

    def test_metadata_is_readable_json(self, model, tmp_path):
        model.save(tmp_path / "bundle")

        metadata = json.loads((tmp_path / "bundle" / "metadata.json").read_text())
        assert metadata["artifact_version"] == 1

    def test_creates_the_output_directory(self, model, tmp_path):
        model.save(tmp_path / "deep" / "nested" / "bundle")

        assert (tmp_path / "deep" / "nested" / "bundle").is_dir()

    def test_loading_a_directory_without_metadata_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()

        with pytest.raises(FileNotFoundError, match="metadata.json"):
            RiskModel.load(tmp_path / "empty")

    def test_refuses_an_artifact_from_a_future_version(self, model, tmp_path):
        model.save(tmp_path / "bundle")
        path = tmp_path / "bundle" / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["artifact_version"] = 999
        path.write_text(json.dumps(metadata))

        with pytest.raises(ValueError, match="not supported"):
            RiskModel.load(tmp_path / "bundle")


class TestScoring:
    def test_score_returns_a_probability_and_a_decision(self, model, frame):
        scored = model.score(frame.head(50))

        assert list(scored.columns) == ["risk_score", "decision"]
        assert scored["risk_score"].between(0, 1).all()

    def test_decisions_are_from_the_declared_set(self, model, frame):
        decisions = set(model.score(frame.head(500))["decision"])

        assert decisions <= {d.value for d in Decision}

    def test_index_is_preserved(self, model, frame):
        sample = frame.tail(20)

        pd.testing.assert_index_equal(model.score(sample).index, sample.index)

    def test_scoring_needs_no_label_column(self, model, frame):
        unlabelled = frame.head(20).drop(columns=[LABEL, "fraud_type"])

        assert len(model.score(unlabelled)) == 20

    def test_unseen_device_platform_degrades_rather_than_crashing(self, model, frame):
        """XGBoost raises on an unseen categorical level; the feature builder
        folds it. Without that, a new platform is an outage."""
        sample = frame.head(50).copy()
        sample["device_os"] = "some_new_os"

        scored = model.score(sample)

        assert np.isfinite(scored["risk_score"]).all()

    def test_a_whole_missing_capability_does_not_crash_scoring(self, model, frame):
        """NFC not deployed yet is a legitimate state."""
        sample = frame.head(50).drop(columns=["nfc_chip_read", "nfc_passive_auth_passed"])

        assert np.isfinite(model.score(sample)["risk_score"]).all()

    def test_a_missing_required_signal_fails_loudly(self, model, frame):
        from idv_risk.features import MissingSignalError

        sample = frame.head(10).drop(columns=["pad_attack_score"])

        with pytest.raises(MissingSignalError):
            model.score(sample)


class TestEvaluate:
    def test_scores_a_labelled_frame(self, model, frame):
        report = evaluate(model, frame.tail(1000))

        assert report["overall"]["n"] == 1000

    def test_reports_the_decision_bands(self, model, frame):
        report = evaluate(model, frame.tail(1000))

        assert len(report["decisions"]) == 3

    def test_reports_a_cost_per_application(self, model, frame):
        report = evaluate(model, frame.tail(1000))

        assert report["cost_per_application"] > 0

    def test_beats_accepting_everything(self, model, frame):
        """The only test that asks whether the model is worth deploying."""
        report = evaluate(model, frame.tail(2000))
        accept_all = report["baseline_costs"]["accept_all"] / report["overall"]["n"]

        assert report["cost_per_application"] < accept_all

    def test_missing_label_is_named_in_the_error(self, model, frame):
        with pytest.raises(KeyError, match=LABEL):
            evaluate(model, frame.head(10).drop(columns=[LABEL]))


class TestSummarise:
    def test_mentions_the_headline_metrics(self, trained):
        text = summarise(trained.test_report)

        for expected in ("PR-AUC", "ISO 30107-3", "inequity rate", "recall by fraud type"):
            assert expected in text

    def test_is_plain_text(self, trained):
        assert "\n" in summarise(trained.test_report)


class TestAblations:
    """What each signal family can and cannot see on its own.

    These train small models on individual stages. They are the empirical
    version of the blind-spot table in ``synthetic.py``, and the reason the
    fusion model earns its place.
    """

    @pytest.fixture(scope="class")
    def ablation_frame(self):
        return generate(GeneratorConfig(n_applications=9_000, seed=21))

    @pytest.fixture(scope="class")
    def ablation_config(self):
        return TrainingConfig(
            model=ModelConfig(n_estimators=60, early_stopping_rounds=10, max_depth=4, n_jobs=2)
        )

    def recall_for(self, result, typology: str) -> float:
        for row in result.test_report["recall_by_fraud_type"]:
            if row["group"] == typology:
                return row["recall"]
        raise AssertionError(f"{typology} absent from the test split")

    @pytest.fixture(scope="class")
    def biometric_only(self, ablation_frame, ablation_config):
        names = feature_names_for_stages(Stage.BIOMETRIC, Stage.PAD)
        return train(ablation_frame, ablation_config, feature_names=names)

    @pytest.fixture(scope="class")
    def graph_only(self, ablation_frame, ablation_config):
        names = feature_names_for_stages(Stage.GRAPH)
        return train(ablation_frame, ablation_config, feature_names=names)

    @pytest.fixture(scope="class")
    def fused(self, ablation_frame, ablation_config):
        return train(ablation_frame, ablation_config)

    def test_biometrics_and_pad_catch_presentation_attacks(self, biometric_only):
        """The case PAD was built for."""
        assert self.recall_for(biometric_only, "presentation") > 0.5

    def test_biometrics_and_pad_are_blind_to_synthetic_identity_rings(self, biometric_only):
        """Every session is genuinely clean; there is nothing for PAD to see.

        This is the single strongest argument for a fusion layer, and the
        reason a stack of certified biometric vendors is not a fraud strategy.
        """
        assert self.recall_for(biometric_only, "synthetic_id") < 0.4

    def test_graph_features_alone_catch_the_rings(self, graph_only):
        assert self.recall_for(graph_only, "synthetic_id") > 0.6

    def test_graph_features_alone_are_blind_to_presentation_attacks(self, graph_only):
        """Symmetry: the graph stage has its own blind spot."""
        assert self.recall_for(graph_only, "presentation") < 0.5

    def test_fusion_beats_both_on_overall_ranking(self, fused, biometric_only, graph_only):
        fused_pr = fused.test_pr_auc

        assert fused_pr > biometric_only.test_pr_auc
        assert fused_pr > graph_only.test_pr_auc

    def test_fusion_catches_every_typology(self, fused):
        for typology in ("presentation", "injection", "document_forgery", "synthetic_id"):
            assert self.recall_for(fused, typology) > 0.4, typology


class TestCli:
    def test_train_writes_a_loadable_bundle(self, tmp_path, capsys):
        exit_code = cli_main(
            ["train", "--rows", "3000", "--seed", "5", "--out", str(tmp_path / "bundle")]
        )

        assert exit_code == 0
        assert RiskModel.load(tmp_path / "bundle") is not None
        assert "PR-AUC" in capsys.readouterr().out

    def test_train_can_write_a_json_report(self, tmp_path):
        cli_main(
            [
                "train",
                "--rows",
                "3000",
                "--seed",
                "5",
                "--out",
                str(tmp_path / "bundle"),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )

        report = json.loads((tmp_path / "report.json").read_text())
        assert "overall" in report

    def test_evaluate_runs_against_a_saved_bundle(self, tmp_path, capsys):
        cli_main(["train", "--rows", "3000", "--seed", "5", "--out", str(tmp_path / "bundle")])
        capsys.readouterr()

        exit_code = cli_main(
            ["evaluate", "--model", str(tmp_path / "bundle"), "--rows", "1500", "--seed", "6"]
        )

        assert exit_code == 0
        assert "PR-AUC" in capsys.readouterr().out

    def test_score_writes_predictions_for_unlabelled_input(self, tmp_path, capsys):
        cli_main(["train", "--rows", "3000", "--seed", "5", "--out", str(tmp_path / "bundle")])
        capsys.readouterr()

        applications = generate(GeneratorConfig(n_applications=100, seed=8))
        applications.drop(columns=[LABEL, "fraud_type"]).to_parquet(tmp_path / "apps.parquet")

        exit_code = cli_main(
            [
                "score",
                "--model",
                str(tmp_path / "bundle"),
                "--input",
                str(tmp_path / "apps.parquet"),
                "--output",
                str(tmp_path / "scored.csv"),
            ]
        )

        assert exit_code == 0
        scored = pd.read_csv(tmp_path / "scored.csv")
        assert len(scored) == 100
        assert set(scored["decision"]) <= {d.value for d in Decision}

    def test_unsupported_input_format_is_rejected(self, tmp_path):
        from idv_risk.cli import load_frame

        with pytest.raises(ValueError, match="unsupported input format"):
            load_frame(tmp_path / "data.xlsx", rows=10, seed=1)

    def test_unknown_subcommand_exits_nonzero(self):
        with pytest.raises(SystemExit):
            cli_main(["nonsense"])
