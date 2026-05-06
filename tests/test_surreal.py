"""Test surreal number recognition and arithmetic."""

from fractions import Fraction

import pytest
from conway_foundations.games.arithmetic import add, clear_arithmetic_cache
from conway_foundations.games.canonical import clear_canonical_caches, equal_as_games
from conway_foundations.games.library import (
    NEG_ONE,
    NEG_THREE,
    NEG_TWO,
    ONE,
    STAR,
    THREE,
    TWO,
    UP,
    ZERO,
)
from conway_foundations.games.outcome import clear_outcome_cache
from conway_foundations.games.surreal import (
    clear_surreal_caches,
    from_number,
    is_number,
    to_number,
)


@pytest.fixture(autouse=True)
def clear_caches():
    clear_canonical_caches()
    clear_arithmetic_cache()
    clear_surreal_caches()
    clear_outcome_cache()
    yield
    clear_canonical_caches()
    clear_arithmetic_cache()
    clear_surreal_caches()
    clear_outcome_cache()


class TestIsNumber:
    def test_zero_is_number(self):
        assert is_number(ZERO)

    def test_one_is_number(self):
        assert is_number(ONE)

    def test_neg_one_is_number(self):
        assert is_number(NEG_ONE)

    def test_star_not_number(self):
        assert not is_number(STAR)

    def test_up_not_number(self):
        assert not is_number(UP)


class TestToNumber:
    def test_zero_value(self):
        assert to_number(ZERO) == Fraction(0)

    def test_one_value(self):
        assert to_number(ONE) == Fraction(1)

    def test_two_value(self):
        assert to_number(TWO) == Fraction(2)

    def test_neg_one_value(self):
        assert to_number(NEG_ONE) == Fraction(-1)

    def test_star_value_none(self):
        assert to_number(STAR) is None

    def test_small_integers_roundtrip(self):
        assert to_number(THREE) == Fraction(3)
        assert to_number(NEG_TWO) == Fraction(-2)
        assert to_number(NEG_THREE) == Fraction(-3)


class TestFromNumber:
    def test_zero_construction(self):
        assert from_number(Fraction(0)) == ZERO

    def test_one_roundtrip(self):
        assert equal_as_games(from_number(Fraction(1)), ONE)

    def test_half_is_number(self):
        half = from_number(Fraction(1, 2))
        assert is_number(half)
        assert to_number(half) == Fraction(1, 2)

    def test_quarter_is_number(self):
        q = from_number(Fraction(1, 4))
        assert is_number(q)
        assert to_number(q) == Fraction(1, 4)


class TestNumberArithmetic:
    def test_one_plus_one_via_numbers(self):
        result = add(ONE, ONE)
        assert to_number(result) == Fraction(2)

    def test_half_plus_half(self):
        half = from_number(Fraction(1, 2))
        result = add(half, half)
        assert to_number(result) == Fraction(1)
