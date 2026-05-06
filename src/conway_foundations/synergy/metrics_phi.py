"""Integrated information Φ_2.5 — Mediano-Rosas-style synergistic MI."""

import numpy as np

from conway_foundations.games.arithmetic import add
from conway_foundations.games.outcome import outcome as outc
from conway_foundations.synergy.metrics import (
    OUTCOME_TO_INT,
    register_metric,
    trajectory_outcomes_independent,
)


class PhiApproxMetric:
    name = "phi_approx"

    def __call__(self, g1, g2, trajectory):
        x1 = trajectory_outcomes_independent(g1, trajectory)
        x2 = trajectory_outcomes_independent(g2, trajectory)
        y = np.array(
            [OUTCOME_TO_INT[outc(add(add(g1, p), g2))] for p in trajectory],
            dtype=int,
        )

        mi_x1_y = self._mi_pair(x1, y)
        mi_x2_y = self._mi_pair(x2, y)
        mi_joint_y = self._mi_joint(x1, x2, y)

        phi_proxy = max(0.0, mi_joint_y - max(mi_x1_y, mi_x2_y))

        return {
            "phi_approx": phi_proxy,
            "mi_x1_y": mi_x1_y,
            "mi_x2_y": mi_x2_y,
            "mi_joint_y": mi_joint_y,
            "sample_size": float(len(trajectory)),
        }

    def _mi_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        joint: dict[tuple[int, int], int] = {}
        mx: dict[int, int] = {}
        my: dict[int, int] = {}
        for a, b in zip(x, y, strict=True):
            ia, ib = int(a), int(b)
            joint[(ia, ib)] = joint.get((ia, ib), 0) + 1
            mx[ia] = mx.get(ia, 0) + 1
            my[ib] = my.get(ib, 0) + 1
        mi = 0.0
        for (a, b), c in joint.items():
            p_ab = c / n
            p_a = mx[a] / n
            p_b = my[b] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += p_ab * np.log2(p_ab / (p_a * p_b))
        return max(0.0, mi)

    def _mi_joint(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
        n = len(x1)
        joint: dict[tuple[tuple[int, int], int], int] = {}
        mxx: dict[tuple[int, int], int] = {}
        my: dict[int, int] = {}
        for a, b, c in zip(x1, x2, y, strict=True):
            ia, ib, ic = int(a), int(b), int(c)
            joint[((ia, ib), ic)] = joint.get(((ia, ib), ic), 0) + 1
            mxx[(ia, ib)] = mxx.get((ia, ib), 0) + 1
            my[ic] = my.get(ic, 0) + 1
        mi = 0.0
        for ((a, b), c), count in joint.items():
            p_abc = count / n
            p_ab = mxx[(a, b)] / n
            p_c = my[c] / n
            if p_abc > 0 and p_ab > 0 and p_c > 0:
                mi += p_abc * np.log2(p_abc / (p_ab * p_c))
        return max(0.0, mi)


register_metric(PhiApproxMetric())
