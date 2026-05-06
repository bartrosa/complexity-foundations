"""Systematic search for non-homomorphism witnesses."""

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from conway_foundations.games.arithmetic import add
from conway_foundations.games.core import Game
from conway_foundations.games.outcome import OutcomeClass, outcome

if TYPE_CHECKING:
    import pandas as pd

OUTCOMES: tuple[OutcomeClass, ...] = ("L", "R", "=", "∥")


@dataclass(frozen=True)
class Witness:
    """A non-homomorphism witness: o(g1)=o(g2) but o(g1+p)≠o(g2+p)."""

    g1_repr: str
    g2_repr: str
    perturbation_repr: str
    base_outcome: OutcomeClass
    compound_outcome_1: OutcomeClass
    compound_outcome_2: OutcomeClass


def find_witnesses(
    games: list[tuple[str, Game]],
    perturbations: list[tuple[str, Game]],
) -> list[Witness]:
    """For each pair (g1, g2) where outcome(g1) == outcome(g2), test
    all perturbations p. Record cases where o(g1+p) != o(g2+p).
    """
    witnesses = []
    for (n1, g1), (n2, g2) in product(games, games):
        if n1 >= n2:
            continue
        o1 = outcome(g1)
        o2 = outcome(g2)
        if o1 != o2:
            continue
        for pname, p in perturbations:
            c1 = outcome(add(g1, p))
            c2 = outcome(add(g2, p))
            if c1 != c2:
                witnesses.append(
                    Witness(
                        g1_repr=n1,
                        g2_repr=n2,
                        perturbation_repr=pname,
                        base_outcome=o1,
                        compound_outcome_1=c1,
                        compound_outcome_2=c2,
                    )
                )
    return witnesses


def witnesses_to_dataframe(witnesses: list[Witness]) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame([w.__dict__ for w in witnesses])
