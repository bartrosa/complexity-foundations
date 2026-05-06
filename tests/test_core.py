"""Test core game outcomes match Conway/Siegel canonical values."""

import pytest
from conway_foundations.games.arithmetic import add, clear_arithmetic_cache, equals, neg
from conway_foundations.games.library import (
    DOWN,
    NEG_ONE,
    ONE,
    STAR,
    STAR_2,
    UP,
    ZERO,
    _integer_game,
)
from conway_foundations.games.outcome import clear_outcome_cache, outcome
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


class TestCanonicalOutcomes:
    """Canonical outcomes from Conway 1976 / Siegel 2013."""

    def test_zero(self):
        assert outcome(ZERO) == "="

    def test_star(self):
        assert outcome(STAR) == "∥"

    def test_one(self):
        assert outcome(ONE) == "L"

    def test_neg_one(self):
        assert outcome(NEG_ONE) == "R"

    def test_up(self):
        assert outcome(UP) == "L"

    def test_down(self):
        assert outcome(DOWN) == "R"

    def test_star_two(self):
        assert outcome(STAR_2) == "∥"


class TestArithmetic:
    """Disjunctive sum and negation."""

    def test_star_plus_star_is_zero(self):
        assert outcome(add(STAR, STAR)) == "="

    def test_one_plus_neg_one(self):
        assert outcome(add(ONE, NEG_ONE)) == "="

    def test_canonical_witness_up_star(self):
        """The Mediano letter witness."""
        assert outcome(add(UP, STAR)) == "∥"

    def test_canonical_witness_one_star(self):
        """The Mediano letter contrast."""
        assert outcome(add(ONE, STAR)) == "L"

    def test_zero_is_identity(self):
        for g in [ZERO, ONE, STAR, UP]:
            assert equals(add(g, ZERO), g)


class TestProperties:
    """Property-based tests via hypothesis."""

    @settings(deadline=2000, suppress_health_check=[HealthCheck.too_slow])
    @given(n=st.integers(min_value=-3, max_value=3))
    def test_negation_swaps_outcomes(self, n):
        g = _integer_game(n)
        o = outcome(g)
        no = outcome(neg(g))
        if o == "L":
            assert no == "R"
        elif o == "R":
            assert no == "L"
        else:
            assert no == o


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear memoization caches between tests to avoid memory blowup."""
    yield
    clear_outcome_cache()
    clear_arithmetic_cache()
