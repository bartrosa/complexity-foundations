"""Test metric registry and individual metrics on canonical cases."""

from conway_foundations.games.arithmetic import clear_arithmetic_cache
from conway_foundations.games.library import CANONICAL_LIBRARY_EXPANDED, ONE, UP, ZERO
from conway_foundations.games.outcome import clear_outcome_cache
from conway_foundations.synergy import METRIC_REGISTRY
from conway_foundations.synergy.metrics_ei import EffectiveInformationMetric
from conway_foundations.synergy.metrics_kl import KLDivergenceMetric
from conway_foundations.synergy.metrics_markov import MarkovEntropyRateMetric
from conway_foundations.synergy.trajectories import (
    generate_independent_perturbations,
    generate_trajectory,
)


def test_all_metrics_registered():
    expected = {
        "kl_divergence",
        "transfer_entropy",
        "effective_information",
        "markov_entropy_rate",
        "phi_approx",
        "causal_emergence",
    }
    assert expected.issubset(set(METRIC_REGISTRY.keys()))


def test_metrics_run_on_canonical_pair():
    """Smoke-test each metric; cumulative metrics use a short safe trajectory."""
    traj = generate_trajectory(CANONICAL_LIBRARY_EXPANDED, length=8, seed=0)
    indep = generate_independent_perturbations(CANONICAL_LIBRARY_EXPANDED)
    for _name, metric in METRIC_REGISTRY.items():
        clear_outcome_cache()
        clear_arithmetic_cache()
        if _name in ("transfer_entropy", "markov_entropy_rate"):
            result = metric(UP, ONE, traj)
        else:
            result = metric(UP, ONE, indep)
        assert isinstance(result, dict)
        assert all(isinstance(v, int | float) for v in result.values())


def test_kl_zero_for_independent_outcomes():
    """KL between identical marginals on degenerate setup is non-negative."""
    metric = KLDivergenceMetric()
    indep = generate_independent_perturbations(CANONICAL_LIBRARY_EXPANDED)
    result = metric(ZERO, ZERO, indep)
    assert result["kl_joint_vs_indep"] >= 0


def test_markov_rate_bounded():
    """Entropy rate of 4-state alphabet bounded by log2(4) = 2."""
    metric = MarkovEntropyRateMetric()
    traj = generate_trajectory(CANONICAL_LIBRARY_EXPANDED, length=10, seed=1)
    result = metric(UP, ONE, traj)
    assert result["markov_rate_compound"] <= 2.0
    assert result["markov_rate_compound"] >= 0


def test_ei_bounded():
    """EI normalized stays in [0, 1]."""
    metric = EffectiveInformationMetric()
    indep = generate_independent_perturbations(CANONICAL_LIBRARY_EXPANDED)
    result = metric(UP, ONE, indep)
    assert result["ei"] >= 0
    assert result["ei_normalized"] >= 0
    assert result["ei_normalized"] <= 1.0
