"""Causal emergence — macro compound vs micro joint structure."""

from itertools import product

import numpy as np

from conway_foundations.games.arithmetic import add
from conway_foundations.games.outcome import outcome as outc
from conway_foundations.synergy.metrics import register_metric


class CausalEmergenceMetric:
    name = "causal_emergence"

    def __call__(self, g1, g2, trajectory):
        micro_inputs = []
        macro_outputs = []
        for p1, p2 in product(trajectory, repeat=2):
            o1 = outc(add(g1, p1))
            o2 = outc(add(g2, p2))
            o_compound = outc(add(add(g1, g2), add(p1, p2)))
            micro_inputs.append((o1, o2))
            macro_outputs.append(o_compound)

        n = len(micro_inputs)
        ei_micro = self._ei(micro_inputs, micro_inputs)
        ei_macro = self._ei(micro_inputs, macro_outputs)

        return {
            "ei_micro": ei_micro,
            "ei_macro": ei_macro,
            "causal_emergence": max(0.0, ei_macro - ei_micro),
            "sample_size": float(n),
        }

    def _ei(self, inputs: list[tuple[str, str]], outputs: list) -> float:
        n = len(inputs)
        joint: dict[tuple[tuple[str, str], str], int] = {}
        mi: dict[tuple[str, str], int] = {}
        mo: dict[str, int] = {}
        for inp, out in zip(inputs, outputs, strict=True):
            joint[(inp, out)] = joint.get((inp, out), 0) + 1
            mi[inp] = mi.get(inp, 0) + 1
            mo[out] = mo.get(out, 0) + 1
        ei = 0.0
        for (inp, out), c in joint.items():
            p_io = c / n
            p_i = mi[inp] / n
            p_o = mo[out] / n
            if p_io > 0 and p_i > 0 and p_o > 0:
                ei += p_io * np.log2(p_io / (p_i * p_o))
        return max(0.0, ei)


register_metric(CausalEmergenceMetric())
