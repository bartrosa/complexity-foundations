"""Conway four-outcome classifier with memoization."""

from functools import cache
from typing import Literal

from conway_foundations.games.core import Game

OutcomeClass = Literal["L", "R", "=", "∥"]


@cache
def left_wins_first(g: Game) -> bool:
    """True iff Left wins moving first.

    Recursive: Left wins by moving to a position where Right (moving
    next) loses.
    """
    return any(not right_wins_first(gl) for gl in g.left)


@cache
def right_wins_first(g: Game) -> bool:
    """True iff Right wins moving first."""
    return any(not left_wins_first(gr) for gr in g.right)


def outcome(g: Game) -> OutcomeClass:
    """Conway outcome classifier o: Games → {L, R, =, ∥}.

    - L: Left wins regardless of who moves first
    - R: Right wins regardless
    - =: Second player wins (P-position, "zero" game)
    - ∥: First player wins (N-position, "fuzzy" game)

    Reference: Conway 1976 ch. 7; Siegel 2013 ch. 2.
    """
    lw = left_wins_first(g)
    rw = right_wins_first(g)
    if lw and not rw:
        return "L"
    if rw and not lw:
        return "R"
    if not lw and not rw:
        return "="
    return "∥"


def clear_outcome_cache():
    """Clear memoization cache. Use between experiments to control memory."""
    left_wins_first.cache_clear()
    right_wins_first.cache_clear()
