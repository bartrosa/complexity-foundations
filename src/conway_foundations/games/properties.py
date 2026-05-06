"""Structural properties of games."""

from functools import cache

from conway_foundations.games.core import Game
from conway_foundations.games.outcome import outcome


@cache
def depth(g: Game) -> int:
    """Birthday: longest path to ZERO.

    {|} has depth 0. Otherwise depth = 1 + max(depth(option)).
    """
    if not g.left and not g.right:
        return 0
    children = list(g.left) + list(g.right)
    return 1 + max((depth(c) for c in children), default=0)


@cache
def num_options(g: Game) -> int:
    return len(g.left) + len(g.right)


@cache
def fuzzy_count(g: Game) -> int:
    """Count of distinct positions in the game DAG whose outcome is fuzzy (∥)."""
    visited: set[Game] = set()
    fuzzy = 0

    def dfs(node: Game) -> None:
        nonlocal fuzzy
        if node in visited:
            return
        visited.add(node)
        if outcome(node) == "∥":
            fuzzy += 1
        for c in list(node.left) + list(node.right):
            dfs(c)

    dfs(g)
    return fuzzy


@cache
def canonical_size(g: Game) -> int:
    """Number of distinct subgames in the DAG."""
    visited: set[Game] = set()

    def dfs(node: Game) -> None:
        if node in visited:
            return
        visited.add(node)
        for c in list(node.left) + list(node.right):
            dfs(c)

    dfs(g)
    return len(visited)


def all_properties(g: Game) -> dict[str, int | str]:
    return {
        "depth": depth(g),
        "num_options": num_options(g),
        "fuzzy_count": fuzzy_count(g),
        "canonical_size": canonical_size(g),
        "outcome": outcome(g),
    }
