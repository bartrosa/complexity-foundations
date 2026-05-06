"""Test canonical form reduction matches expected canonical values."""

import pytest
from conway_foundations.games.arithmetic import add, clear_arithmetic_cache, equals, neg
from conway_foundations.games.canonical import (
    canonicalize,
    clear_canonical_caches,
    equal_as_games,
    geq,
    leq,
)
from conway_foundations.games.core import Game
from conway_foundations.games.library import (
    CANONICAL_LIBRARY_EXPANDED,
    DOWN,
    NEG_ONE,
    ONE,
    STAR,
    TWO,
    UP,
    ZERO,
)
from conway_foundations.games.outcome import clear_outcome_cache, outcome


@pytest.fixture(autouse=True)
def clear_caches():
    clear_canonical_caches()
    clear_arithmetic_cache()
    clear_outcome_cache()
    yield
    clear_canonical_caches()
    clear_arithmetic_cache()
    clear_outcome_cache()


class TestBasicCanonical:
    def test_zero_is_canonical(self):
        assert canonicalize(ZERO) == ZERO

    def test_star_is_canonical(self):
        assert canonicalize(STAR) == STAR

    def test_one_is_canonical(self):
        assert canonicalize(ONE) == ONE


class TestComparison:
    def test_one_geq_zero(self):
        assert geq(ONE, ZERO)
        assert not geq(ZERO, ONE)

    def test_two_geq_one(self):
        assert geq(TWO, ONE)
        assert geq(TWO, ZERO)

    def test_neg_one_lt_zero(self):
        assert leq(NEG_ONE, ZERO)
        assert not leq(ZERO, NEG_ONE)

    def test_zero_equals_zero(self):
        assert equal_as_games(ZERO, ZERO)

    def test_star_not_equal_zero(self):
        assert not equal_as_games(STAR, ZERO)


class TestDominationRemoval:
    def test_dominated_left_option_removed(self):
        """{0, 1 | } canonicalizes to {1 | } — 1 dominates 0 on the left."""
        raw = Game(frozenset({ZERO, ONE}), frozenset())
        canon = canonicalize(raw)
        assert ONE in canon.left
        assert ZERO not in canon.left


class TestSumCanonicalForm:
    def test_one_plus_one_equals_two(self):
        result = add(ONE, ONE)
        assert equal_as_games(result, TWO)

    def test_one_plus_neg_one_equals_zero(self):
        result = add(ONE, NEG_ONE)
        assert equal_as_games(result, ZERO)

    def test_star_plus_star_equals_zero(self):
        result = add(STAR, STAR)
        assert equal_as_games(result, ZERO)

    def test_zero_is_additive_identity(self):
        for g in [ZERO, ONE, NEG_ONE, STAR, UP, DOWN]:
            assert equal_as_games(add(g, ZERO), g)


class TestOutcomePreservedAfterCanonicalization:
    """Canonicalization must not change outcomes."""

    def test_outcomes_preserved_for_library(self):
        for name, g in CANONICAL_LIBRARY_EXPANDED:
            outcome_before = outcome(g)
            canon = canonicalize(g)
            outcome_after = outcome(canon)
            assert outcome_before == outcome_after, f"{name}: {outcome_before} → {outcome_after}"


class TestPerformance:
    def test_repeated_sum_does_not_explode(self):
        """Repeated STAR sums stay small in canonical form."""
        g = ZERO
        for _ in range(10):
            g = add(g, STAR)
        canon = canonicalize(g)
        from conway_foundations.games.properties import canonical_size

        assert canonical_size(canon) < 100

    def test_canonical_nonhomomorphism_witness_still_holds(self):
        assert outcome(add(UP, STAR)) == "∥"
        assert outcome(add(ONE, STAR)) == "L"


class TestEqualsMatchesOutcome:
    def test_equals_agrees_with_zero_sum(self):
        """equals(g,h) agrees with outcome(add(g, neg(h))) for spot checks."""
        for a in [ZERO, ONE, STAR, UP]:
            for b in [ZERO, ONE, STAR]:
                via_order = equals(a, b)
                via_outcome = outcome(add(a, neg(b))) == "="
                assert via_order == via_outcome, (a, b)
