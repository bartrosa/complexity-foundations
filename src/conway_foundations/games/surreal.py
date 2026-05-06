"""Recognize when a game is a (dyadic rational) surreal number.

A game is a number iff every left option < every right option, and every
option is itself a number.

For numbers: G = {a | b} with a < b yields the simplest number strictly
between a and b (dyadic rationals in finite cases).

This module provides:
- is_number(g): bool
- to_number(g): rational value if is_number(g), else None
- from_number(q): construct a standard game representing dyadic rational q
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import cache

from conway_foundations.games.canonical import leq
from conway_foundations.games.core import Game


@cache
def is_number(g: Game) -> bool:
    """True iff G is a (surreal) number (all options numeric; all L < all R)."""
    for opt in list(g.left) + list(g.right):
        if not is_number(opt):
            return False
    for gl in g.left:
        for gr in g.right:
            if not leq(gl, gr) or leq(gr, gl):
                return False
    return True


@cache
def to_number(g: Game) -> Fraction | None:
    """If G is a number, return its rational value; else None."""
    if not is_number(g):
        return None

    if not g.left and not g.right:
        return Fraction(0)

    left_vals: list[Fraction] = [v for v in (to_number(gl) for gl in g.left) if v is not None]
    right_vals: list[Fraction] = [v for v in (to_number(gr) for gr in g.right) if v is not None]

    if not left_vals:
        if not right_vals:
            return Fraction(0)
        min_r = min(right_vals)
        if min_r > 0:
            return Fraction(0)
        k = min_r.numerator // min_r.denominator
        return Fraction(k - 1)

    if not right_vals:
        max_l = max(left_vals)
        if max_l < 0:
            return Fraction(0)
        k = max_l.numerator // max_l.denominator
        return Fraction(k + 1)

    max_l = max(left_vals)
    min_r = min(right_vals)
    return _simplest_number_between(max_l, min_r)


def _simplest_number_between(a: Fraction, b: Fraction) -> Fraction:
    """Simplest dyadic rational strictly between a and b."""
    if a >= b:
        msg = f"Need a < b, got {a} >= {b}"
        raise ValueError(msg)

    lo = int(math.floor(float(a)) + 1)
    if a < lo < b:
        return Fraction(lo)

    for k in range(1, 64):
        denom = 2**k
        num_low = math.floor(float(a * denom)) + 1
        num_high = math.ceil(float(b * denom)) - 1
        for num in range(num_low, num_high + 1):
            cand = Fraction(num, denom)
            if a < cand < b:
                return cand

    msg = f"Could not find dyadic rational between {a} and {b}"
    raise RuntimeError(msg)


def _is_dyadic(q: Fraction) -> bool:
    d = q.denominator
    return d & (d - 1) == 0


@cache
def from_number(q: Fraction) -> Game:
    """Construct a standard game for dyadic rational q."""
    if q == 0:
        return Game()

    if not _is_dyadic(q):
        msg = f"{q} is not a dyadic rational"
        raise ValueError(msg)

    if q.denominator == 1:
        n = q.numerator
        if n > 0:
            return Game(frozenset({from_number(Fraction(n - 1))}), frozenset())
        return Game(frozenset(), frozenset({from_number(Fraction(n + 1))}))

    k = q.denominator.bit_length() - 1
    eps = Fraction(1, 2**k)
    return Game(
        frozenset({from_number(q - eps)}),
        frozenset({from_number(q + eps)}),
    )


def add_as_numbers(g: Game, h: Game) -> Game | None:
    """If both are numbers, return from_number(g + h); else None."""
    qg = to_number(g)
    qh = to_number(h)
    if qg is None or qh is None:
        return None
    total = qg + qh
    return from_number(total)


def clear_surreal_caches() -> None:
    is_number.cache_clear()
    to_number.cache_clear()
    from_number.cache_clear()
