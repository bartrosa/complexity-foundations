"""Effective information — causal structure at coarse-graining."""

from itertools import product

import numpy as np

from conway_foundations.games.arithmetic import add
from conway_foundations.games.outcome import outcome as outc
from conway_foundations.synergy.metrics import register_metric


class EffectiveInformationMetric:
    """EI of the mapping (o(g1+p1), o(g2+p2)) → o(g1+p1+g2+p2) under uniform pairs.

    NOTE: `num_distinct_compounds` is computed for diagnostics but typically has
    very low variance across pairs (few distinct compound outcomes in a 4-class
    space). It is excluded from primary structural analysis in
    `synergy_library` but kept in CSV output.
    """

    name = "effective_information"

    def __call__(self, g1, g2, trajectory):
        universe = list(trajectory)

        all_compounds = []
        for p1, p2 in product(universe, repeat=2):
            compound = add(add(g1, p1), add(g2, p2))
            all_compounds.append((outc(p1), outc(p2), outc(compound)))

        n = len(all_compounds)
        joint: dict[tuple[tuple[str, str], str], int] = {}
        marg_input: dict[tuple[str, str], int] = {}
        marg_output: dict[str, int] = {}
        for inp1, inp2, outp in all_compounds:
            inp = (inp1, inp2)
            joint[(inp, outp)] = joint.get((inp, outp), 0) + 1
            marg_input[inp] = marg_input.get(inp, 0) + 1
            marg_output[outp] = marg_output.get(outp, 0) + 1

        ei = 0.0
        for (inp, outp), count in joint.items():
            p_joint = count / n
            p_inp = marg_input[inp] / n
            p_out = marg_output[outp] / n
            if p_joint > 0 and p_inp > 0 and p_out > 0:
                ei += p_joint * np.log2(p_joint / (p_inp * p_out))

        max_ei = np.log2(max(1, len(marg_output)))
        determinism = ei / max_ei if max_ei > 0 else 0.0

        return {
            "ei": max(0.0, ei),
            "ei_normalized": max(0.0, min(1.0, determinism)),
            "num_distinct_compounds": float(len(marg_output)),
            "sample_size": float(n),
        }


register_metric(EffectiveInformationMetric())
