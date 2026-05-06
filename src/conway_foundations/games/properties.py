"""Numeric properties of games (birthday / depth)."""

from functools import cache

from conway_foundations.games.core import Game


@cache
def depth(g: Game) -> int:
    """Short birthday (minimal ordinal depth).

    {|} has depth 0. Otherwise depth = 1 + max(depth(option)) over all
    left and right options.

    Reference: Conway 1976, ONAG (birthday).
    """
    if not g.left and not g.right:
        return 0
    depths = [depth(x) for x in g.left] + [depth(x) for x in g.right]
    return 1 + max(depths)
