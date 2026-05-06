"""DPLL solver instrumented with a backtrack counter."""

from __future__ import annotations

import random

from conway_foundations.topology.sat import SATInstance


def _clause_satisfied(a: list[bool | None], clause: tuple[tuple[int, bool], ...]) -> bool:
    return any(a[v] is not None and (a[v] != neg) for v, neg in clause)


def _clause_conflict(a: list[bool | None], clause: tuple[tuple[int, bool], ...]) -> bool:
    for v, neg in clause:
        if a[v] is None:
            return False
        if a[v] != neg:
            return False
    return True


def dpll(
    phi: SATInstance,
    var_order: list[int] | None = None,
) -> tuple[bool, int]:
    """Return (satisfiable, num_backtracks) for vanilla DPLL over variable order."""
    n = phi.n_vars
    if var_order is None:
        var_order = list(range(n))
    backtracks = [0]

    def solve(a: list[bool | None], depth: int) -> bool:
        if any(_clause_conflict(a, c) for c in phi.clauses):
            return False
        if all(_clause_satisfied(a, c) for c in phi.clauses):
            return True
        if depth >= len(var_order):
            return False
        var = var_order[depth]
        for val in (True, False):
            na = a.copy()
            na[var] = val
            if solve(na, depth + 1):
                return True
            backtracks[0] += 1
        return False

    initial: list[bool | None] = [None] * n
    return solve(initial, 0), backtracks[0]


def measure_hardness(
    phi: SATInstance,
    n_orderings: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Measure DPLL difficulty across random variable orderings."""
    rng = random.Random(seed)
    counts: list[int] = []
    for _ in range(n_orderings):
        order = list(range(phi.n_vars))
        rng.shuffle(order)
        _, bt = dpll(phi, order)
        counts.append(bt)
    counts.sort()
    mid = len(counts) // 2
    return {
        "mean_backtracks": float(sum(counts) / len(counts)),
        "median_backtracks": float(counts[mid]),
        "min_backtracks": float(counts[0]),
        "max_backtracks": float(counts[-1]),
    }
