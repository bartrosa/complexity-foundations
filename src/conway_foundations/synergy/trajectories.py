"""Generate perturbation trajectories for metric computation."""

from __future__ import annotations

import random

from conway_foundations.games.core import Game

# Cumulative disjunctive sums blow up quickly if perturbations include integer chains
# or multi-option stars; keep trajectory sampling on a **minimal** tractable pool.
# Omit UP/DOWN from cumulative draws: repeated `add` with ↑/↓ makes intermediate sums huge.
DEFAULT_TRAJECTORY_COMPONENT_NAMES: frozenset[str] = frozenset(
    {
        "ZERO",
        "ONE",
        "NEG_ONE",
        "STAR",
    }
)


def game_pool_by_names(
    library: list[tuple[str, Game]],
    names: frozenset[str] | set[str] | None = None,
) -> list[Game]:
    """Subset of library games whose symbolic names are in ``names``."""
    chosen = DEFAULT_TRAJECTORY_COMPONENT_NAMES if names is None else names
    return [g for label, g in library if label in chosen]


def generate_trajectory(
    library: list[tuple[str, Game]],
    length: int = 20,
    seed: int = 42,
    *,
    games_subset: list[Game] | None = None,
) -> list[Game]:
    """Sample perturbations with replacement to build a trajectory.

    Parameters
    ----------
    games_subset
        Pool to sample from. Defaults to :func:`game_pool_by_names` so cumulative
        sums along the trajectory stay computationally tractable.
    """
    rng = random.Random(seed)
    pool = games_subset if games_subset is not None else game_pool_by_names(library)
    if not pool:
        pool = [g for _, g in library]
    return [rng.choice(pool) for _ in range(length)]


def generate_independent_perturbations(
    library: list[tuple[str, Game]],
) -> list[Game]:
    """Full library as independent perturbations (one sample each)."""
    return [g for _, g in library]
