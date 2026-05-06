"""Disjunctive sum, negation, equality on games."""

from functools import cache

from conway_foundations.games.core import Game
from conway_foundations.games.outcome import outcome


@cache
def add(g: Game, h: Game) -> Game:
    """Disjunctive sum G + H. Conway 1976 ch. 7.

    Either player moves in one component, leaving the other unchanged.
    """
    new_left = frozenset(add(gl, h) for gl in g.left) | frozenset(add(g, hl) for hl in h.left)
    new_right = frozenset(add(gr, h) for gr in g.right) | frozenset(add(g, hr) for hr in h.right)
    return Game(new_left, new_right)


@cache
def neg(g: Game) -> Game:
    """Negation -G: swap Left and Right roles recursively."""
    return Game(
        frozenset(neg(gr) for gr in g.right),
        frozenset(neg(gl) for gl in g.left),
    )


def equals(g: Game, h: Game) -> bool:
    """G = H iff G + (-H) is a P-position (second-player win, '=' outcome)."""
    return outcome(add(g, neg(h))) == "="


def clear_arithmetic_cache():
    """Clear memoization cache. Use between experiments."""
    add.cache_clear()
    neg.cache_clear()
