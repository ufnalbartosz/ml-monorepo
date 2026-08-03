"""The cost-based decision policy and the evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idv_risk.config import CostConfig
from idv_risk.decision import (
    Decision,
    Thresholds,
    baseline_costs,
    decision_report,
    expected_cost,
    optimise_thresholds,
)
from idv_risk.metrics import (
    apcer_bpcer,
    inequity_rate,
    recall_at_fpr,
    recall_by_group,
    score_report,
    subgroup_report,
)


@pytest.fixture
def separable():
    """A well-behaved score distribution: fraud high, genuine low."""
    rng = np.random.default_rng(0)
    genuine = rng.beta(1, 20, 950)
    fraud = rng.beta(6, 3, 50)
    probabilities = np.concatenate([genuine, fraud])
    labels = np.concatenate([np.zeros(950), np.ones(50)])
    return probabilities, labels


class TestThresholds:
    def test_bands_are_assigned_correctly(self):
        thresholds = Thresholds(accept_below=0.2, reject_at_or_above=0.8)

        decisions = thresholds.decide(np.array([0.1, 0.5, 0.9]))

        assert list(decisions) == [
            Decision.ACCEPT.value,
            Decision.STEP_UP.value,
            Decision.REJECT.value,
        ]

    def test_boundaries_are_half_open(self):
        thresholds = Thresholds(0.2, 0.8)

        assert thresholds.decide(np.array([0.2]))[0] == Decision.STEP_UP.value
        assert thresholds.decide(np.array([0.8]))[0] == Decision.REJECT.value

    def test_equal_boundaries_give_an_empty_step_up_band(self):
        thresholds = Thresholds(0.5, 0.5)

        decisions = thresholds.decide(np.array([0.4, 0.6]))

        assert Decision.STEP_UP.value not in set(decisions)

    def test_rejects_inverted_boundaries(self):
        with pytest.raises(ValueError, match="must not exceed"):
            Thresholds(accept_below=0.9, reject_at_or_above=0.1)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_out_of_range_boundaries(self, bad):
        with pytest.raises(ValueError, match="must be in"):
            Thresholds(accept_below=bad, reject_at_or_above=1.0)

    def test_round_trips_through_a_dict(self):
        thresholds = Thresholds(0.25, 0.75)

        assert Thresholds.from_dict(thresholds.to_dict()) == thresholds


class TestExpectedCost:
    def test_a_perfect_policy_costs_nothing(self):
        cost = CostConfig()
        probabilities = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])

        assert expected_cost(probabilities, labels, Thresholds(0.5, 0.5), cost) == 0.0

    def test_accepting_fraud_costs_the_false_accept_price(self):
        cost = CostConfig(false_accept=500.0)

        total = expected_cost(np.array([0.0]), np.array([1.0]), Thresholds(0.5, 0.5), cost)

        assert total == pytest.approx(500.0)

    def test_rejecting_a_genuine_applicant_costs_the_false_reject_price(self):
        cost = CostConfig(false_reject=60.0)

        total = expected_cost(np.array([1.0]), np.array([0.0]), Thresholds(0.5, 0.5), cost)

        assert total == pytest.approx(60.0)

    def test_step_up_is_not_free(self):
        """Otherwise the optimiser routes everything to a review team that
        does not exist."""
        cost = CostConfig(step_up=12.0, step_up_miss_rate=0.1, step_up_abandon_rate=0.15)
        everything_stepped = Thresholds(0.0, 1.0)

        fraud = expected_cost(np.array([0.5]), np.array([1.0]), everything_stepped, cost)
        genuine = expected_cost(np.array([0.5]), np.array([0.0]), everything_stepped, cost)

        assert fraud == pytest.approx(12.0 + 0.1 * 500.0)
        assert genuine == pytest.approx(12.0 + 0.15 * 60.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="disagree on length"):
            expected_cost(np.zeros(3), np.zeros(2), Thresholds(0.5, 0.5), CostConfig())

    def test_rejects_an_empty_population(self):
        with pytest.raises(ValueError, match="empty population"):
            expected_cost(np.array([]), np.array([]), Thresholds(0.5, 0.5), CostConfig())


class TestOptimiseThresholds:
    def test_beats_every_trivial_policy(self, separable):
        probabilities, labels = separable
        cost = CostConfig()

        thresholds = optimise_thresholds(probabilities, labels, cost)
        chosen = expected_cost(probabilities, labels, thresholds, cost)

        assert chosen < min(baseline_costs(labels, cost).values())

    def test_respects_the_step_up_cap(self, separable):
        probabilities, labels = separable

        thresholds = optimise_thresholds(probabilities, labels, CostConfig(), max_step_up_rate=0.05)

        step_up_rate = np.mean(thresholds.decide(probabilities) == Decision.STEP_UP.value)
        assert step_up_rate <= 0.05 + 1e-9

    def test_an_uncapped_search_can_use_more_review(self, separable):
        probabilities, labels = separable

        capped = optimise_thresholds(probabilities, labels, CostConfig(), max_step_up_rate=0.02)
        uncapped = optimise_thresholds(probabilities, labels, CostConfig(), max_step_up_rate=None)

        capped_cost = expected_cost(probabilities, labels, capped, CostConfig())
        uncapped_cost = expected_cost(probabilities, labels, uncapped, CostConfig())
        assert uncapped_cost <= capped_cost + 1e-9

    def test_expensive_false_rejects_push_the_reject_boundary_up(self, separable):
        """The policy has to follow the costs, or the costs are decoration."""
        probabilities, labels = separable

        cautious = optimise_thresholds(
            probabilities, labels, CostConfig(false_reject=5000.0), max_step_up_rate=None
        )
        aggressive = optimise_thresholds(
            probabilities, labels, CostConfig(false_reject=1.0), max_step_up_rate=None
        )

        assert cautious.reject_at_or_above >= aggressive.reject_at_or_above

    def test_expensive_fraud_pushes_the_accept_boundary_down(self, separable):
        probabilities, labels = separable

        strict = optimise_thresholds(
            probabilities, labels, CostConfig(false_accept=100_000.0), max_step_up_rate=None
        )
        lax = optimise_thresholds(
            probabilities, labels, CostConfig(false_accept=61.0), max_step_up_rate=None
        )

        assert strict.accept_below <= lax.accept_below

    def test_rejects_an_impossible_cap(self, separable):
        probabilities, labels = separable

        with pytest.raises(ValueError, match="no threshold pair"):
            optimise_thresholds(probabilities, labels, CostConfig(), max_step_up_rate=-1.0)

    def test_rejects_an_empty_population(self):
        with pytest.raises(ValueError, match="empty population"):
            optimise_thresholds(np.array([]), np.array([]), CostConfig())


class TestDecisionReport:
    def test_has_one_row_per_band(self, separable):
        probabilities, labels = separable

        report = decision_report(probabilities, labels, Thresholds(0.2, 0.8))

        assert list(report["decision"]) == ["accept", "step_up", "reject"]

    def test_shares_sum_to_one(self, separable):
        probabilities, labels = separable

        report = decision_report(probabilities, labels, Thresholds(0.2, 0.8))

        assert report["share"].sum() == pytest.approx(1.0)

    def test_fraud_rate_rises_across_the_bands(self, separable):
        """If the step-up band's fraud rate matches the accept band's, the band
        is buying friction and no security."""
        probabilities, labels = separable

        report = decision_report(probabilities, labels, Thresholds(0.2, 0.8))
        rates = report.set_index("decision")["fraud_rate"]

        assert rates["accept"] < rates["step_up"] < rates["reject"]

    def test_carries_the_expected_cost(self, separable):
        probabilities, labels = separable

        report = decision_report(probabilities, labels, Thresholds(0.2, 0.8))

        assert report.attrs["expected_cost"] > 0
        assert report.attrs["cost_per_application"] == pytest.approx(
            report.attrs["expected_cost"] / len(labels)
        )


class TestBaselineCosts:
    def test_accept_all_costs_every_fraud(self):
        labels = np.array([1.0] * 10 + [0.0] * 90)

        assert baseline_costs(labels, CostConfig(false_accept=500.0))["accept_all"] == 5000.0

    def test_reject_all_costs_every_genuine_applicant(self):
        labels = np.array([1.0] * 10 + [0.0] * 90)

        assert baseline_costs(labels, CostConfig(false_reject=60.0))["reject_all"] == 5400.0

    def test_reports_all_three_baselines(self):
        costs = baseline_costs(np.array([1.0, 0.0]), CostConfig())

        assert set(costs) == {"accept_all", "reject_all", "step_up_all"}


class TestRecallAtFpr:
    def test_a_perfect_ranker_catches_everything(self, separable):
        probabilities = np.concatenate([np.zeros(90), np.ones(10)])
        labels = np.concatenate([np.zeros(90), np.ones(10)])

        assert recall_at_fpr(probabilities, labels, 0.01) == pytest.approx(1.0)

    def test_a_random_ranker_catches_little(self):
        rng = np.random.default_rng(0)
        labels = np.concatenate([np.zeros(950), np.ones(50)])

        assert recall_at_fpr(rng.random(1000), labels, 0.01) < 0.2

    def test_is_non_decreasing_in_the_budget(self, separable):
        probabilities, labels = separable

        at_low = recall_at_fpr(probabilities, labels, 0.001)
        at_high = recall_at_fpr(probabilities, labels, 0.05)

        assert at_high >= at_low

    def test_single_class_input_is_undefined_not_an_error(self):
        assert np.isnan(recall_at_fpr(np.array([0.5, 0.6]), np.zeros(2), 0.01))

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5])
    def test_rejects_an_impossible_budget(self, bad, separable):
        probabilities, labels = separable

        with pytest.raises(ValueError, match="target_fpr"):
            recall_at_fpr(probabilities, labels, bad)


class TestScoreReport:
    def test_reports_the_base_rate(self, separable):
        probabilities, labels = separable

        assert score_report(probabilities, labels).base_rate == pytest.approx(0.05)

    def test_pr_auc_beats_the_base_rate_for_a_useful_model(self, separable):
        probabilities, labels = separable
        report = score_report(probabilities, labels)

        assert report.pr_auc > report.base_rate * 5

    def test_single_class_input_yields_nan_rather_than_raising(self):
        report = score_report(np.array([0.1, 0.2]), np.zeros(2))

        assert np.isnan(report.pr_auc)
        assert np.isnan(report.roc_auc)

    def test_dict_form_flattens_the_recall_targets(self, separable):
        probabilities, labels = separable

        payload = score_report(probabilities, labels).to_dict()

        assert "recall_at_fpr_0.01" in payload

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="disagree on length"):
            score_report(np.zeros(3), np.zeros(2))


class TestApcerBpcer:
    def test_a_perfect_system_scores_zero_on_both(self):
        probabilities = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])

        assert apcer_bpcer(probabilities, labels, 0.5) == (0.0, 0.0)

    def test_apcer_counts_attacks_that_got_through(self):
        probabilities = np.array([0.1, 0.9, 0.1])
        labels = np.array([1.0, 1.0, 0.0])

        apcer, _ = apcer_bpcer(probabilities, labels, 0.5)

        assert apcer == pytest.approx(0.5)

    def test_bpcer_counts_genuine_users_rejected(self):
        probabilities = np.array([0.9, 0.1, 0.9])
        labels = np.array([0.0, 0.0, 1.0])

        _, bpcer = apcer_bpcer(probabilities, labels, 0.5)

        assert bpcer == pytest.approx(0.5)

    def test_they_trade_off_against_each_other(self, separable):
        probabilities, labels = separable

        strict = apcer_bpcer(probabilities, labels, 0.1)
        lenient = apcer_bpcer(probabilities, labels, 0.9)

        assert strict[0] <= lenient[0]  # APCER
        assert strict[1] >= lenient[1]  # BPCER

    def test_needs_both_classes(self):
        with pytest.raises(ValueError, match="both attack and bona fide"):
            apcer_bpcer(np.array([0.5, 0.6]), np.zeros(2), 0.5)


class TestSubgroupReport:
    @pytest.fixture
    def biased(self):
        """One group gets a much higher false-positive rate than the other."""
        rng = np.random.default_rng(1)
        n = 600
        groups = pd.Series(["A"] * n + ["B"] * n)
        labels = np.tile(np.concatenate([np.zeros(n - 60), np.ones(60)]), 2)
        good = rng.beta(1, 30, n)
        bad = rng.beta(1, 6, n)
        return np.concatenate([good, bad]), labels, groups

    def test_one_row_per_group(self, biased):
        probabilities, labels, groups = biased

        assert len(subgroup_report(probabilities, labels, groups, 0.3)) == 2

    def test_detects_a_differential_that_was_planted(self, biased):
        probabilities, labels, groups = biased

        table = subgroup_report(probabilities, labels, groups, 0.3).set_index("group")

        assert table.loc["B", "fpr"] > table.loc["A", "fpr"]

    def test_marks_groups_with_too_few_cases_of_either_class(self):
        groups = pd.Series(["big"] * 200 + ["tiny"] * 4)
        labels = np.concatenate([np.tile([0, 1], 100), [0, 1, 0, 1]])
        probabilities = np.linspace(0, 1, 204)

        table = subgroup_report(probabilities, labels, groups, 0.5).set_index("group")

        assert table.loc["big", "sufficient_data"]
        assert not table.loc["tiny", "sufficient_data"]

    def test_smoothed_rates_are_never_zero(self):
        """A raw zero makes the inequity ratio infinite."""
        groups = pd.Series(["A"] * 100)
        labels = np.concatenate([np.zeros(50), np.ones(50)])
        probabilities = np.concatenate([np.zeros(50), np.ones(50)])

        table = subgroup_report(probabilities, labels, groups, 0.5)

        assert (table["fpr"] == 0.0).all()
        assert (table["fpr_smoothed"] > 0.0).all()


class TestInequityRate:
    def test_identical_groups_score_one(self):
        table = pd.DataFrame(
            {
                "group": ["A", "B"],
                "fpr_smoothed": [0.02, 0.02],
                "fnr_smoothed": [0.10, 0.10],
                "sufficient_data": [True, True],
            }
        )

        assert inequity_rate(table) == pytest.approx(1.0)

    def test_doubling_both_error_rates_quadruples_the_score(self):
        table = pd.DataFrame(
            {
                "group": ["A", "B"],
                "fpr_smoothed": [0.02, 0.04],
                "fnr_smoothed": [0.10, 0.20],
                "sufficient_data": [True, True],
            }
        )

        assert inequity_rate(table) == pytest.approx(4.0)

    def test_ignores_groups_with_insufficient_data(self):
        table = pd.DataFrame(
            {
                "group": ["A", "B", "noise"],
                "fpr_smoothed": [0.02, 0.02, 0.9],
                "fnr_smoothed": [0.10, 0.10, 0.9],
                "sufficient_data": [True, True, False],
            }
        )

        assert inequity_rate(table) == pytest.approx(1.0)

    def test_is_undefined_with_fewer_than_two_usable_groups(self):
        table = pd.DataFrame(
            {
                "group": ["A"],
                "fpr_smoothed": [0.02],
                "fnr_smoothed": [0.10],
                "sufficient_data": [True],
            }
        )

        assert np.isnan(inequity_rate(table))

    def test_is_finite_even_when_a_group_made_no_errors(self):
        groups = pd.Series(["A"] * 200 + ["B"] * 200)
        labels = np.tile(np.concatenate([np.zeros(100), np.ones(100)]), 2)
        # Group A is perfect, group B is not.
        probabilities = np.concatenate(
            [np.zeros(100), np.ones(100), np.full(100, 0.9), np.ones(100)]
        )

        table = subgroup_report(probabilities, labels, groups, 0.5)

        assert np.isfinite(inequity_rate(table))


class TestRecallByGroup:
    def test_reports_recall_per_typology(self):
        groups = pd.Series(["a", "a", "b", "b"])
        labels = np.array([1, 1, 1, 1])
        probabilities = np.array([0.9, 0.9, 0.9, 0.1])

        table = recall_by_group(probabilities, labels, groups, 0.5).set_index("group")

        assert table.loc["a", "recall"] == pytest.approx(1.0)
        assert table.loc["b", "recall"] == pytest.approx(0.5)

    def test_groups_with_no_positives_report_nan(self):
        groups = pd.Series(["genuine", "genuine"])
        labels = np.array([0, 0])

        table = recall_by_group(np.array([0.1, 0.2]), labels, groups, 0.5)

        assert np.isnan(table.loc[0, "recall"])

    def test_works_when_the_group_series_has_a_non_zero_based_index(self):
        """`_build_report` passes a slice of the original frame."""
        groups = pd.Series(["a", "b"], index=[5000, 5001])

        table = recall_by_group(np.array([0.9, 0.1]), np.array([1, 1]), groups, 0.5)

        assert len(table) == 2
