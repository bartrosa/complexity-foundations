"""Check if outcome is an F-algebra morphism for F(X) = P(X) × P(X).

If outcome were an F-algebra morphism, there would exist φ: 4×4→4
such that for all G, H: o(G+H) = φ(o(G), o(H)). We exhaustively
test candidate φ on the library.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, cast

from conway_foundations.games.arithmetic import add
from conway_foundations.games.core import Game
from conway_foundations.games.outcome import OutcomeClass, outcome
from conway_foundations.utils.parallel import parallel_map

OUTCOMES: tuple[OutcomeClass, ...] = ("L", "R", "=", "∥")


@dataclass
class MorphismCheckResult:
    is_morphism: bool
    tested_phi_count: int
    valid_phi_count: int
    counterexamples_per_phi: dict[int, int]
    sample_counterexamples: list[dict[str, Any]]


def candidate_phi_functions(
    commutative_only: bool = True,
    zero_identity: bool = True,
) -> list[dict[tuple[OutcomeClass, OutcomeClass], OutcomeClass]]:
    """Generate candidate functions φ: 4×4 → 4.

    Reductions to keep search tractable:
    - commutative: φ(a,b) = φ(b,a) since + is commutative
    - zero_identity: φ('=', x) = x since 0 is the identity for +

    Without reductions: 4^16 = 4.3 billion functions.
    With both: 4^6 = 4096 candidates (free pairs among non-identity outcomes).
    """
    pairs_raw = list(product(OUTCOMES, repeat=2))
    if commutative_only:
        pairs_raw = [(a, b) for a, b in pairs_raw if a <= b]
    if zero_identity:
        pairs_raw = [(a, b) for a, b in pairs_raw if a != "=" and b != "="]
    pairs = cast(list[tuple[OutcomeClass, OutcomeClass]], pairs_raw)

    phis: list[dict[tuple[OutcomeClass, OutcomeClass], OutcomeClass]] = []
    for assignment in product(OUTCOMES, repeat=len(pairs)):
        phi: dict[tuple[OutcomeClass, OutcomeClass], OutcomeClass] = dict(
            zip(pairs, assignment, strict=True),
        )
        if zero_identity:
            for x in OUTCOMES:
                phi[("=", x)] = x
                phi[(x, "=")] = x
        if commutative_only:
            full_phi = dict(phi)
            for (a, b), v in phi.items():
                full_phi[(b, a)] = v
            phi = full_phi
        phis.append(phi)
    return phis


def check_phi_against_pairs(
    phi: dict[tuple[OutcomeClass, OutcomeClass], OutcomeClass],
    pairs_with_compounds: list[tuple[OutcomeClass, OutcomeClass, OutcomeClass]],
) -> tuple[bool, list[tuple[OutcomeClass, OutcomeClass, OutcomeClass | None, OutcomeClass]]]:
    """Test a single φ. Returns (works, counterexamples).

    pairs_with_compounds: list of (o(G), o(H), o(G+H)).
    """
    counterexamples: list[tuple[OutcomeClass, OutcomeClass, OutcomeClass | None, OutcomeClass]] = []
    for o_g, o_h, o_compound in pairs_with_compounds:
        predicted = phi.get((o_g, o_h))
        if predicted != o_compound:
            counterexamples.append((o_g, o_h, predicted, o_compound))
    return len(counterexamples) == 0, counterexamples


def check_outcome_is_morphism(
    games: list[Game],
    n_jobs: int = -1,
) -> MorphismCheckResult:
    """Exhaustively test whether any φ predicts compound outcomes on all pairs."""
    pairs = list(product(games, games))

    def _compute_triple(pair: tuple[Game, Game]) -> tuple[OutcomeClass, OutcomeClass, OutcomeClass]:
        g, h = pair
        return (outcome(g), outcome(h), outcome(add(g, h)))

    triples = parallel_map(_compute_triple, pairs, n_jobs=n_jobs, verbose=0)

    phis = candidate_phi_functions(commutative_only=True, zero_identity=True)

    valid_count = 0
    counterexamples_per_phi: dict[int, int] = {}
    sample_counterexamples: list[dict[str, Any]] = []

    for i, phi in enumerate(phis):
        works, ce = check_phi_against_pairs(phi, triples)
        if works:
            valid_count += 1
        counterexamples_per_phi[i] = len(ce)
        if i < 3:
            sample_counterexamples.append({"phi_idx": i, "counterexamples": ce[:5]})

    return MorphismCheckResult(
        is_morphism=(valid_count > 0),
        tested_phi_count=len(phis),
        valid_phi_count=valid_count,
        counterexamples_per_phi=counterexamples_per_phi,
        sample_counterexamples=sample_counterexamples,
    )
