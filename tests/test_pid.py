"""Test PID computation correctness."""

import numpy as np
from conway_foundations.games.library import ONE, STAR, UP, ZERO
from conway_foundations.synergy.pid import (
    compute_pid,
    game_pair_to_distribution,
    joint_distribution,
    joint_mutual_information,
    mutual_information,
    williams_beer_pid,
)


def test_xor_synergy():
    """XOR has synergy ≈ 1 bit (canonical PID test case)."""
    rng = np.random.default_rng(42)
    n = 4000
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y = x1 ^ x2
    result = williams_beer_pid(x1, x2, y)
    assert result["synergy"] > 0.7, f"Got {result['synergy']:.3f}"


def test_independent_zero_synergy():
    """X1 ⊥ X2, Y = X1 has zero synergy."""
    rng = np.random.default_rng(43)
    n = 4000
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y = x1.copy()
    result = williams_beer_pid(x1, x2, y)
    assert result["synergy"] < 0.1, f"Got {result['synergy']:.3f}"


def test_redundant_dominates():
    """Y = X1 = X2 has redundancy ≈ 1 bit, no synergy."""
    rng = np.random.default_rng(44)
    n = 4000
    x1 = rng.integers(0, 2, n)
    x2 = x1.copy()
    y = x1.copy()
    result = williams_beer_pid(x1, x2, y)
    assert result["redundant"] > 0.7
    assert result["synergy"] < 0.1


def test_pid_for_canonical_witness_runs():
    """Smoke test on the canonical non-homomorphism witness pair."""
    library = [ZERO, ONE, UP, STAR]
    x1, x2, y = game_pair_to_distribution(UP, ONE, library)
    result = compute_pid(x1, x2, y, method="williams-beer")
    assert result["synergy"] >= 0
    assert result["redundant"] >= 0
    assert result["mi_total"] >= 0


def test_mi_consistency():
    """I((X1,X2);Y) >= I(X1;Y) and >= I(X2;Y)."""
    rng = np.random.default_rng(45)
    n = 1000
    x1 = rng.integers(0, 2, n)
    x2 = rng.integers(0, 2, n)
    y = x1 ^ x2
    joint = joint_distribution(x1, x2, y)
    mi_x1y = mutual_information(joint, 0, 2)
    mi_x2y = mutual_information(joint, 1, 2)
    mi_jointy = joint_mutual_information(joint)
    assert mi_jointy + 1e-6 >= mi_x1y
    assert mi_jointy + 1e-6 >= mi_x2y
