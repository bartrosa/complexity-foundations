"""Partial Information Decomposition for game-pair outcomes.

Given a pair (G1, G2) and perturbations [p_1, ..., p_n], build empirical
joint samples (X1, X2, Y) from outcomes of G1+p_i, G2+p_i and apply PID.
"""

from typing import Literal, cast

import numpy as np

from conway_foundations.games.arithmetic import add
from conway_foundations.games.core import Game
from conway_foundations.games.outcome import outcome

PIDMethod = Literal["williams-beer", "broja", "ppid"]
OUTCOME_TO_INT = {"L": 0, "R": 1, "=": 2, "∥": 3}


def game_pair_to_distribution(
    g1: Game,
    g2: Game,
    perturbations: list[Game],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X1, X2, Y) integer arrays from empirical perturbations."""
    x1, x2, y = [], [], []
    for p in perturbations:
        o1 = OUTCOME_TO_INT[outcome(add(g1, p))]
        o2 = OUTCOME_TO_INT[outcome(add(g2, p))]
        x1.append(o1)
        x2.append(o2)
        y.append(o1 * 4 + o2)
    return np.array(x1), np.array(x2), np.array(y)


def joint_distribution(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
) -> dict[tuple[int, int, int], float]:
    """Empirical joint P(X1, X2, Y) as dict."""
    n = len(x1)
    counts: dict[tuple[int, int, int], int] = {}
    for a, b, c in zip(x1, x2, y, strict=True):
        key = (int(a), int(b), int(c))
        counts[key] = counts.get(key, 0) + 1
    return {k: v / n for k, v in counts.items()}


def _marginal(joint: dict[tuple[int, int, int], float], var_idx: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, p in joint.items():
        v = key[var_idx]
        out[v] = out.get(v, 0.0) + p
    return out


def _pairwise_marginal(
    joint: dict[tuple[int, int, int], float],
    idx_a: int,
    idx_b: int,
) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for key, p in joint.items():
        v = (key[idx_a], key[idx_b])
        out[v] = out.get(v, 0.0) + p
    return out


def mutual_information(
    joint: dict[tuple[int, int, int], float],
    var_a: int,
    var_b: int,
) -> float:
    """I(A; B) from full joint over (X1, X2, Y)."""
    p_a = _marginal(joint, var_a)
    p_b = _marginal(joint, var_b)
    p_ab = _pairwise_marginal(joint, var_a, var_b)
    mi = 0.0
    for (a, b), p in p_ab.items():
        if p > 0 and p_a[a] > 0 and p_b[b] > 0:
            mi += p * np.log2(p / (p_a[a] * p_b[b]))
    return max(0.0, mi)


def joint_mutual_information(joint: dict[tuple[int, int, int], float]) -> float:
    """I((X1,X2); Y)."""
    p_y = _marginal(joint, 2)
    p_xx = _pairwise_marginal(joint, 0, 1)
    p_xxy: dict[tuple[tuple[int, int], int], float] = {}
    for key, p in joint.items():
        a, b, c = key
        k = ((a, b), c)
        p_xxy[k] = p_xxy.get(k, 0.0) + p
    mi = 0.0
    for ((a, b), c), p in p_xxy.items():
        if p > 0 and p_y[c] > 0 and p_xx[(a, b)] > 0:
            mi += p * np.log2(p / (p_xx[(a, b)] * p_y[c]))
    return max(0.0, mi)


def williams_beer_pid(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Williams-Beer 2010 PID for two sources onto Y."""
    joint = joint_distribution(x1, x2, y)
    p_y = _marginal(joint, 2)

    redundant = 0.0
    for y_val, p_y_val in p_y.items():
        if p_y_val == 0:
            continue
        spec_info = []
        for source_idx in [0, 1]:
            p_source = _marginal(joint, source_idx)
            si = 0.0
            for key, _p in joint.items():
                if key[2] != y_val:
                    continue
                src = key[source_idx]
                mass_y = sum(pj for kj, pj in joint.items() if kj[2] == y_val)
                if mass_y == 0:
                    continue
                p_src_given_y = (
                    sum(pj for kj, pj in joint.items() if kj[2] == y_val and kj[source_idx] == src)
                    / mass_y
                )
                if p_src_given_y > 0 and p_source.get(src, 0) > 0:
                    si += p_src_given_y * np.log2(p_src_given_y / p_source[src])
            spec_info.append(si)
        redundant += p_y_val * min(spec_info)
    redundant = max(0.0, redundant)

    mi_x1_y = mutual_information(joint, 0, 2)
    mi_x2_y = mutual_information(joint, 1, 2)
    mi_joint_y = joint_mutual_information(joint)

    unique_x1 = max(0.0, mi_x1_y - redundant)
    unique_x2 = max(0.0, mi_x2_y - redundant)
    synergy = max(0.0, mi_joint_y - redundant - unique_x1 - unique_x2)

    return {
        "synergy": synergy,
        "redundant": redundant,
        "unique_x1": unique_x1,
        "unique_x2": unique_x2,
        "mi_total": mi_joint_y,
    }


def _pid_via_dit(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    method: PIDMethod,
) -> dict[str, float]:
    """Use dit for BROJA / PPID when available."""
    import dit
    from dit.pid import PID_BROJA, PID_PROJ

    joint = joint_distribution(x1, x2, y)
    outcomes_dit: list[tuple[int, int, int]] = []
    probs_dit: list[float] = []
    for (a, b, c), p in joint.items():
        outcomes_dit.append((int(a), int(b), int(c)))
        probs_dit.append(p)
    d = dit.Distribution(outcomes_dit, probs_dit)
    d.set_rv_names(["X1", "X2", "Y"])

    if method == "broja":
        pid = PID_BROJA(d, [("X1",), ("X2",)], "Y")
    elif method == "ppid":
        pid = PID_PROJ(d, [("X1",), ("X2",)], "Y")
    else:
        raise ValueError(method)

    return {
        "synergy": float(pid.get_pi(((("X1",), ("X2",)),))),
        "redundant": float(pid.get_pi((("X1",), ("X2",)))),
        "unique_x1": float(pid.get_pi((("X1",),))),
        "unique_x2": float(pid.get_pi((("X2",),))),
        "mi_total": float(d.mutual_information(["X1", "X2"], ["Y"])),
    }


def compute_pid(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    method: PIDMethod = "williams-beer",
) -> dict[str, float | str]:
    """Compute PID; BROJA/PPID fall back to Williams-Beer if dit fails."""
    if method == "williams-beer":
        return cast(dict[str, float | str], williams_beer_pid(x1, x2, y))
    try:
        return cast(dict[str, float | str], _pid_via_dit(x1, x2, y, method))
    except Exception as e:
        result: dict[str, float | str] = dict(williams_beer_pid(x1, x2, y))
        result["fallback_reason"] = str(type(e).__name__)
        return result


def pid_all_methods(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray
) -> dict[str, dict[str, float | str]]:
    return {
        "williams-beer": compute_pid(x1, x2, y, "williams-beer"),
        "broja": compute_pid(x1, x2, y, "broja"),
        "ppid": compute_pid(x1, x2, y, "ppid"),
    }
