"""Canonical form reduction for Conway games.

Two-stage reduction (Conway 1976 ch.7, Siegel 2013 §2.3):
1. Domination: in g.left, if option A ≥ option B, B can be removed.
   Symmetrically for g.right (if A ≤ B, A can be removed).
2. Reversibility: if option A ∈ g.left has a counter-option A' ∈ A.right
   with A' ≤ g, then A is "reversible through A'" — replace A with the
   left-options of A'.

After both reductions iterated to fixpoint, the game is in canonical
form. Two games are equal iff their canonical forms are structurally
identical.

Reference: Siegel, "Combinatorial Game Theory", Theorem 2.9 and
algorithm in §2.3.
"""

from __future__ import annotations

from functools import cache

from conway_foundations.games.core import Game


@cache
def leq(g: Game, h: Game) -> bool:
    """G ≤ H (Conway partial order).

    Recursive definition: G ≤ H iff no G^L satisfies H ≤ G^L and no H^R
    satisfies H^R ≤ G.
    """
    if any(leq(h, gl) for gl in g.left):
        return False
    return all(not leq(hr, g) for hr in h.right)


@cache
def lt_or_fuzzy(g: Game, h: Game) -> bool:
    """G < H or G ∥ H: not (H ≤ G)."""
    return not leq(h, g)


def geq(g: Game, h: Game) -> bool:
    """G ≥ H iff H ≤ G."""
    return leq(h, g)


@cache
def equal_as_games(g: Game, h: Game) -> bool:
    """G = H as combinatorial games (game equality, not structural identity)."""
    return leq(g, h) and leq(h, g)


@cache
def remove_dominated_left(options: frozenset[Game]) -> frozenset[Game]:
    """Remove options A ∈ left such that ∃ B ∈ left with B ≥ A and B ≠ A.

    Left prefers higher values; dominated-from-above options are redundant.
    Equivalent duplicates keep the first occurrence in iteration order.
    """
    keep: list[Game] = []
    options_list = list(options)
    for i, a in enumerate(options_list):
        dominated = False
        for j, b in enumerate(options_list):
            if i == j:
                continue
            if geq(b, a) and not equal_as_games(a, b):
                dominated = True
                break
            if equal_as_games(a, b) and j < i:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return frozenset(keep)


@cache
def remove_dominated_right(options: frozenset[Game]) -> frozenset[Game]:
    """Remove options A ∈ right such that ∃ B ∈ right with B ≤ A and B ≠ A."""
    keep: list[Game] = []
    options_list = list(options)
    for i, a in enumerate(options_list):
        dominated = False
        for j, b in enumerate(options_list):
            if i == j:
                continue
            if leq(b, a) and not equal_as_games(a, b):
                dominated = True
                break
            if equal_as_games(a, b) and j < i:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return frozenset(keep)


@cache
def bypass_reversible_left(options: frozenset[Game], parent: Game) -> frozenset[Game]:
    """If A ∈ left has A' ∈ A.right with A' ≤ parent, replace A by A'.left."""
    new_options: list[Game] = []
    changed = False
    for a in options:
        bypassed = False
        for a_prime in a.right:
            if leq(a_prime, parent):
                new_options.extend(a_prime.left)
                bypassed = True
                changed = True
                break
        if not bypassed:
            new_options.append(a)
    return frozenset(new_options) if changed else options


@cache
def bypass_reversible_right(options: frozenset[Game], parent: Game) -> frozenset[Game]:
    """If A ∈ right has A' ∈ A.left with A' ≥ parent, replace A by A'.right."""
    new_options: list[Game] = []
    changed = False
    for a in options:
        bypassed = False
        for a_prime in a.left:
            if geq(a_prime, parent):
                new_options.extend(a_prime.right)
                bypassed = True
                changed = True
                break
        if not bypassed:
            new_options.append(a)
    return frozenset(new_options) if changed else options


@cache
def canonicalize(g: Game) -> Game:
    """Return a canonical form of g (domination + reversibility to fixpoint)."""
    canon_left = frozenset(canonicalize(gl) for gl in g.left)
    canon_right = frozenset(canonicalize(gr) for gr in g.right)

    prev_left: frozenset[Game] | None = None
    prev_right: frozenset[Game] | None = None
    cur_left, cur_right = canon_left, canon_right

    iterations = 0
    while (prev_left, prev_right) != (cur_left, cur_right):
        prev_left, prev_right = cur_left, cur_right

        cur_left = remove_dominated_left(cur_left)
        cur_right = remove_dominated_right(cur_right)
        candidate = Game(cur_left, cur_right)

        cur_left = bypass_reversible_left(cur_left, candidate)
        cur_right = bypass_reversible_right(cur_right, candidate)

        iterations += 1
        if iterations > 50:
            break

    return Game(cur_left, cur_right)


def clear_canonical_caches() -> None:
    """Clear all canonicalization caches."""
    leq.cache_clear()
    lt_or_fuzzy.cache_clear()
    equal_as_games.cache_clear()
    remove_dominated_left.cache_clear()
    remove_dominated_right.cache_clear()
    bypass_reversible_left.cache_clear()
    bypass_reversible_right.cache_clear()
    canonicalize.cache_clear()
