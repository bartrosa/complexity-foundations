"""Disjunctive sum, negation, equality on games — canonicalized."""

from __future__ import annotations

from functools import cache

from conway_foundations.games.canonical import canonicalize, equal_as_games
from conway_foundations.games.core import Game


@cache
def add_raw(g: Game, h: Game) -> Game:
    """Disjunctive sum **without** canonical form reduction (internal)."""
    new_left = frozenset(add_raw(gl, h) for gl in g.left) | frozenset(
        add_raw(g, hl) for hl in h.left
    )
    new_right = frozenset(add_raw(gr, h) for gr in g.right) | frozenset(
        add_raw(g, hr) for hr in h.right
    )
    return Game(new_left, new_right)


@cache
def add(g: Game, h: Game) -> Game:
    """Disjunctive sum G + H in canonical form.

    Surreal numbers short-circuit via dyadic arithmetic; otherwise raw sum
    is canonicalized so memoization stays effective on finite libraries.
    """
    from conway_foundations.games.surreal import add_as_numbers

    number_result = add_as_numbers(g, h)
    if number_result is not None:
        return number_result
    return canonicalize(add_raw(g, h))


@cache
def neg(g: Game) -> Game:
    """Negation -G in canonical form."""
    raw = Game(
        frozenset(neg(gr) for gr in g.right),
        frozenset(neg(gl) for gl in g.left),
    )
    return canonicalize(raw)


def equals(g: Game, h: Game) -> bool:
    """G = H as games (via Conway partial order)."""
    return equal_as_games(g, h)


def clear_arithmetic_cache() -> None:
    """Clear memoization caches. Use between experiments."""
    add.cache_clear()
    add_raw.cache_clear()
    neg.cache_clear()
