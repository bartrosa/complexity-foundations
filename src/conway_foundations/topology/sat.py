"""Random 3-SAT instances and assignment evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SATInstance:
    n_vars: int
    clauses: tuple[tuple[tuple[int, bool], ...], ...]


def random_3sat(n: int, m: int, seed: int = 42) -> SATInstance:
    """Generate random 3-SAT with n variables and m clauses.

    Each clause is three literals ``(var_index, is_negative)`` where
    ``is_negative`` is True iff the literal is negated.
    """
    rng = random.Random(seed)
    if n < 3:
        msg = f"Need n >= 3 for 3-SAT, got {n}"
        raise ValueError(msg)
    clauses: list[tuple[tuple[int, bool], ...]] = []
    for _ in range(m):
        vars_in_clause = rng.sample(range(n), 3)
        clause = tuple((v, rng.choice([True, False])) for v in vars_in_clause)
        clauses.append(clause)
    return SATInstance(n_vars=n, clauses=tuple(clauses))


def all_assignments(n: int) -> list[tuple[bool, ...]]:
    """Enumerate all 2^n boolean assignments. Limited to n <= 16."""
    if n > 16:
        msg = f"Too many vars for exhaustive enum: {n}"
        raise ValueError(msg)
    return [tuple(bool((i >> k) & 1) for k in range(n)) for i in range(2**n)]


def satisfied_clauses(phi: SATInstance, assignment: tuple[bool, ...]) -> frozenset[int]:
    """Indices of clauses satisfied by the assignment."""
    sat: set[int] = set()
    for i, clause in enumerate(phi.clauses):
        if any(assignment[v] != negated for v, negated in clause):
            sat.add(i)
    return frozenset(sat)


def is_satisfiable_brute(phi: SATInstance) -> bool:
    """Brute-force satisfiability — for testing only; exponential in n."""
    target = len(phi.clauses)
    return any(len(satisfied_clauses(phi, a)) == target for a in all_assignments(phi.n_vars))
