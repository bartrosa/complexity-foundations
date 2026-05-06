"""Unified registry of information-theoretic metrics on game pairs."""

from typing import Protocol

import numpy as np

from conway_foundations.games.arithmetic import add
from conway_foundations.games.core import Game
from conway_foundations.games.outcome import outcome

OUTCOME_TO_INT = {"L": 0, "R": 1, "=": 2, "∥": 3}


class GameMetric(Protocol):
    name: str

    def __call__(
        self,
        g1: Game,
        g2: Game,
        trajectory: list[Game],
    ) -> dict[str, float]: ...


def encode_outcome(g: Game) -> int:
    return OUTCOME_TO_INT[outcome(g)]


def trajectory_outcomes(g: Game, trajectory: list[Game]) -> np.ndarray:
    """Apply perturbations cumulatively: [g+p1, (g+p1)+p2, ...]."""
    current = g
    out = []
    for p in trajectory:
        current = add(current, p)
        out.append(encode_outcome(current))
    return np.array(out, dtype=int)


def trajectory_outcomes_independent(g: Game, trajectory: list[Game]) -> np.ndarray:
    """Apply perturbations independently: [g+p1, g+p2, g+p3, ...]."""
    return np.array([encode_outcome(add(g, p)) for p in trajectory], dtype=int)


METRIC_REGISTRY: dict[str, GameMetric] = {}


def register_metric(metric: GameMetric) -> GameMetric:
    METRIC_REGISTRY[metric.name] = metric
    return metric
