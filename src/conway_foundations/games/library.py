"""Canonical games library.

References:
- Conway 1976, On Numbers and Games (ONAG)
- Siegel 2013, Combinatorial Game Theory (AMS GSM 146)
"""

from conway_foundations.games.core import Game

ZERO = Game()
"""0 = {|}. ONAG p.4. Outcome: =. Birthday: 0."""


def _integer_game(n: int) -> Game:
    """Construct integer n as Conway game."""
    if n == 0:
        return ZERO
    if n > 0:
        return Game(frozenset({_integer_game(n - 1)}), frozenset())
    return Game(frozenset(), frozenset({_integer_game(n + 1)}))


ONE = Game(frozenset({ZERO}), frozenset())
"""1 = {0|}. ONAG p.10. Outcome: L. Birthday: 1."""

TWO = Game(frozenset({ONE}), frozenset())
"""2 = {1|}. ONAG p.11. Outcome: L. Birthday: 2."""

THREE = _integer_game(3)
"""3 = {2|}. Outcome: L."""

NEG_ONE = Game(frozenset(), frozenset({ZERO}))
"""-1 = {|0}. ONAG p.10. Outcome: R."""

NEG_TWO = Game(frozenset(), frozenset({NEG_ONE}))
"""-2 = {|-1}. Outcome: R."""

NEG_THREE = _integer_game(-3)
"""-3. Outcome: R."""

STAR = Game(frozenset({ZERO}), frozenset({ZERO}))
"""* = {0|0}. ONAG p.71. Outcome: ∥. Smallest fuzzy game."""

UP = Game(frozenset({ZERO}), frozenset({STAR}))
"""↑ = {0|*}. Siegel p.62. Outcome: L (L wins regardless of who starts)."""

DOWN = Game(frozenset({STAR}), frozenset({ZERO}))
"""↓ = {*|0}. Siegel p.62. Outcome: R."""

STAR_2 = Game(frozenset({ZERO, STAR}), frozenset({ZERO, STAR}))
"""*2 = {0,*|0,*}. ONAG p.124. Outcome: ∥."""

STAR_3 = Game(frozenset({ZERO, STAR, STAR_2}), frozenset({ZERO, STAR, STAR_2}))
"""*3 = {0,*,*2|0,*,*2}. Outcome: ∥."""

STAR_4 = Game(
    frozenset({ZERO, STAR, STAR_2, STAR_3}),
    frozenset({ZERO, STAR, STAR_2, STAR_3}),
)
"""*4. Outcome: ∥."""

UP_UP = Game(frozenset({ZERO}), frozenset({UP}))
"""↑↑ = {0|↑}."""

_UP_STAR_CHILD = Game(frozenset({ZERO}), frozenset({STAR}))
DOUBLE_UP_STAR = Game(frozenset({ZERO}), frozenset({_UP_STAR_CHILD}))
"""⇑* — double-up plus star variant (not in default expanded list)."""


def switch(n: int) -> Game:
    """Switch ±n = {n|-n}. ONAG ch.5. Outcome: ∥ (hot game)."""
    pos = _integer_game(n)
    neg = _integer_game(-n)
    return Game(frozenset({pos}), frozenset({neg}))


SWITCH_HALF = Game(frozenset({ONE}), frozenset({ZERO}))
"""Asymmetric switch-like game {1|0}."""

SWITCH_NEG_HALF = Game(frozenset({ZERO}), frozenset({NEG_ONE}))
"""Asymmetric switch-like game {0|-1}."""

CANONICAL_LIBRARY = [ZERO, ONE, TWO, NEG_ONE, NEG_TWO, STAR, STAR_2, UP, DOWN]
"""Default library for systematic experiments."""

CANONICAL_LIBRARY_EXPANDED: list[tuple[str, Game]] = [
    ("ZERO", ZERO),
    ("ONE", ONE),
    ("TWO", TWO),
    ("THREE", THREE),
    ("NEG_ONE", NEG_ONE),
    ("NEG_TWO", NEG_TWO),
    ("NEG_THREE", NEG_THREE),
    ("STAR", STAR),
    ("STAR_2", STAR_2),
    ("STAR_3", STAR_3),
    ("STAR_4", STAR_4),
    ("UP", UP),
    ("DOWN", DOWN),
    ("UP_UP", UP_UP),
    ("SWITCH_1", switch(1)),
    ("SWITCH_2", switch(2)),
    ("SWITCH_3", switch(3)),
    ("SWITCH_HALF", SWITCH_HALF),
    ("SWITCH_NEG_HALF", SWITCH_NEG_HALF),
]
