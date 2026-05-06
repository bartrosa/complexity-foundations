"""Canonical games library.

References:
- Conway 1976, On Numbers and Games (ONAG)
- Siegel 2013, Combinatorial Game Theory (AMS GSM 146)
"""

from conway_foundations.games.core import Game

ZERO = Game()
"""0 = {|}. ONAG p.4. Outcome: =. Birthday: 0."""

ONE = Game(frozenset({ZERO}), frozenset())
"""1 = {0|}. ONAG p.10. Outcome: L. Birthday: 1."""

TWO = Game(frozenset({ONE}), frozenset())
"""2 = {1|}. ONAG p.11. Outcome: L. Birthday: 2."""

NEG_ONE = Game(frozenset(), frozenset({ZERO}))
"""-1 = {|0}. ONAG p.10. Outcome: R."""

NEG_TWO = Game(frozenset(), frozenset({NEG_ONE}))
"""-2 = {|-1}. Outcome: R."""

STAR = Game(frozenset({ZERO}), frozenset({ZERO}))
"""* = {0|0}. ONAG p.71. Outcome: ∥. Smallest fuzzy game."""

UP = Game(frozenset({ZERO}), frozenset({STAR}))
"""↑ = {0|*}. Siegel p.62. Outcome: L (L wins regardless of who starts)."""

DOWN = Game(frozenset({STAR}), frozenset({ZERO}))
"""↓ = {*|0}. Siegel p.62. Outcome: R."""

STAR_2 = Game(frozenset({ZERO, STAR}), frozenset({ZERO, STAR}))
"""*2 = {0,*|0,*}. ONAG p.124. Outcome: ∥."""

# Switches: ±n = {n|-n}


def switch(n: int) -> Game:
    """Switch ±n = {n|-n}. ONAG ch.5. Outcome: ∥ (hot game)."""
    pos = _integer_game(n)
    neg = _integer_game(-n)
    return Game(frozenset({pos}), frozenset({neg}))


def _integer_game(n: int) -> Game:
    """Construct integer n as Conway game."""
    if n == 0:
        return ZERO
    if n > 0:
        return Game(frozenset({_integer_game(n - 1)}), frozenset())
    return Game(frozenset(), frozenset({_integer_game(n + 1)}))


CANONICAL_LIBRARY = [ZERO, ONE, TWO, NEG_ONE, NEG_TWO, STAR, STAR_2, UP, DOWN]
"""Default library for systematic experiments."""
