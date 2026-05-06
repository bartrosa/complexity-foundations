"""Causal emergence: macro effective information minus max part-level EI."""

from itertools import product

import numpy as np

from conway_foundations.synergy.metrics import register_metric


class CausalEmergenceMetric:
    """CE = EI_macro − max(EI_micro_g1, EI_micro_g2).

    EI_micro_gi: how informative perturbation outcomes are about o(gi + pi).
    EI_macro: how informative joint perturbations are about o(g1 + g2 + p1 + p2).

    Positive CE means whole-system dynamics carry more causal structure than
    the parts alone (under this coarse proxy).
    """

    name = "causal_emergence"

    def __call__(self, g1, g2, trajectory):
        ei_micro_g1 = self._ei_for_single(g1, trajectory)
        ei_micro_g2 = self._ei_for_single(g2, trajectory)
        ei_macro = self._ei_for_pair(g1, g2, trajectory)

        ce = ei_macro - max(ei_micro_g1, ei_micro_g2)

        return {
            "ei_micro_g1": ei_micro_g1,
            "ei_micro_g2": ei_micro_g2,
            "ei_macro": ei_macro,
            "causal_emergence": ce,
            "sample_size": float(len(trajectory) ** 2),
        }

    def _ei_for_single(self, g, trajectory):
        """EI of mapping (outcome of perturbation → outcome of g + perturbation)."""
        from conway_foundations.games.arithmetic import add
        from conway_foundations.games.outcome import outcome as outc

        inputs = [outc(p) for p in trajectory]
        outputs = [outc(add(g, p)) for p in trajectory]
        return self._mutual_information(inputs, outputs)

    def _ei_for_pair(self, g1, g2, trajectory):
        """EI of (joint perturbation outcomes → compound outcome)."""
        from conway_foundations.games.arithmetic import add
        from conway_foundations.games.outcome import outcome as outc

        inputs = []
        outputs = []
        for p1, p2 in product(trajectory, repeat=2):
            inputs.append((outc(p1), outc(p2)))
            outputs.append(outc(add(add(g1, g2), add(p1, p2))))
        return self._mutual_information(inputs, outputs)

    @staticmethod
    def _mutual_information(inputs, outputs):
        """Plain MI(input, output) by counting."""
        n = len(inputs)
        joint: dict = {}
        mi: dict = {}
        mo: dict = {}
        for inp, out in zip(inputs, outputs, strict=True):
            joint[(inp, out)] = joint.get((inp, out), 0) + 1
            mi[inp] = mi.get(inp, 0) + 1
            mo[out] = mo.get(out, 0) + 1
        result = 0.0
        for (inp, out), c in joint.items():
            p_io = c / n
            p_i = mi[inp] / n
            p_o = mo[out] / n
            if p_io > 0 and p_i > 0 and p_o > 0:
                result += p_io * np.log2(p_io / (p_i * p_o))
        return max(0.0, result)


register_metric(CausalEmergenceMetric())
