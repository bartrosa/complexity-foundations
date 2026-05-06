"""KL divergence: joint outcome distribution vs independent product."""

import numpy as np

from conway_foundations.synergy.metrics import (
    register_metric,
    trajectory_outcomes_independent,
)


class KLDivergenceMetric:
    name = "kl_divergence"

    def __call__(self, g1, g2, trajectory):
        x1 = trajectory_outcomes_independent(g1, trajectory)
        x2 = trajectory_outcomes_independent(g2, trajectory)

        n = len(x1)
        joint: dict[tuple[int, int], int] = {}
        marginal_1: dict[int, int] = {}
        marginal_2: dict[int, int] = {}
        for a, b in zip(x1, x2, strict=True):
            ia, ib = int(a), int(b)
            joint[(ia, ib)] = joint.get((ia, ib), 0) + 1
            marginal_1[ia] = marginal_1.get(ia, 0) + 1
            marginal_2[ib] = marginal_2.get(ib, 0) + 1

        kl = 0.0
        for (a, b), count in joint.items():
            p_joint = count / n
            p_indep = (marginal_1[a] / n) * (marginal_2[b] / n)
            if p_joint > 0 and p_indep > 0:
                kl += p_joint * np.log2(p_joint / p_indep)

        return {
            "kl_joint_vs_indep": max(0.0, kl),
            "sample_size": float(n),
        }


register_metric(KLDivergenceMetric())
