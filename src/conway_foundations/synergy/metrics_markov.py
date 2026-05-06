"""Markov chain entropy rate of perturbation trajectories."""

import numpy as np

from conway_foundations.games.arithmetic import add
from conway_foundations.games.outcome import outcome as outc
from conway_foundations.synergy.metrics import OUTCOME_TO_INT, register_metric, trajectory_outcomes


class MarkovEntropyRateMetric:
    """Entropy rate H(o_{t+1} | o_t) under cumulative perturbation dynamics."""

    name = "markov_entropy_rate"

    def __call__(self, g1, g2, trajectory):
        traj = list(trajectory)
        x1_traj = trajectory_outcomes(g1, traj)
        x2_traj = trajectory_outcomes(g2, traj)

        rate_1 = self._entropy_rate(x1_traj)
        rate_2 = self._entropy_rate(x2_traj)

        compound = []
        cur1, cur2 = g1, g2
        for p in traj:
            cur1 = add(cur1, p)
            cur2 = add(cur2, p)
            compound.append(OUTCOME_TO_INT[outc(add(cur1, cur2))])
        compound_arr = np.array(compound, dtype=int)
        rate_compound = self._entropy_rate(compound_arr)

        return {
            "markov_rate_g1": rate_1,
            "markov_rate_g2": rate_2,
            "markov_rate_compound": rate_compound,
            "markov_excess_compound": max(0.0, rate_compound - max(rate_1, rate_2)),
            "sample_size": float(len(traj)),
        }

    def _entropy_rate(self, seq: np.ndarray) -> float:
        """H(X_{t+1} | X_t) by counting transitions."""
        if len(seq) < 2:
            return 0.0
        transitions: dict[tuple[int, int], int] = {}
        from_counts: dict[int, int] = {}
        for i in range(len(seq) - 1):
            f, t = int(seq[i]), int(seq[i + 1])
            transitions[(f, t)] = transitions.get((f, t), 0) + 1
            from_counts[f] = from_counts.get(f, 0) + 1

        rate = 0.0
        n = len(seq) - 1
        for (f, _), c in transitions.items():
            p_ft = c / n
            p_t_given_f = c / from_counts[f]
            if p_t_given_f > 0:
                rate -= p_ft * np.log2(p_t_given_f)
        return max(0.0, rate)


register_metric(MarkovEntropyRateMetric())
