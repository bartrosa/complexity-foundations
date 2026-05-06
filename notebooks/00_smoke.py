# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Smoke test — infrastructure works
#
# Hypothesis: All canonical game outcomes match Conway 1976 / Siegel 2013.
#
# Kill criterion: Any canonical outcome differs from reference.
#
# Survive criterion: All match, parallel infrastructure detected.
#

# %% tags=["parameters"]
# Papermill parameters - override via papermill -p key value
random_seed = 42
output_dir = 'data'
figures_dir = 'figures'
n_jobs = -1

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from conway_foundations.utils.parallel import (
    detect_compute_backend,
    parallel_map,
)

np.random.seed(random_seed)
output_dir = Path(output_dir)
figures_dir = Path(figures_dir)
output_dir.mkdir(exist_ok=True, parents=True)
figures_dir.mkdir(exist_ok=True, parents=True)

print("Compute backend:", detect_compute_backend())

# %% [markdown]
# ## Computation
#

# %%
from conway_foundations.games.arithmetic import add
from conway_foundations.games.library import (
    DOWN,
    NEG_ONE,
    NEG_TWO,
    ONE,
    STAR,
    STAR_2,
    TWO,
    UP,
    ZERO,
)
from conway_foundations.games.outcome import outcome

games = {
    'ZERO': ZERO, 'ONE': ONE, 'TWO': TWO,
    'NEG_ONE': NEG_ONE, 'NEG_TWO': NEG_TWO,
    'STAR': STAR, 'STAR_2': STAR_2,
    'UP': UP, 'DOWN': DOWN,
}
expected = {
    'ZERO': '=', 'ONE': 'L', 'TWO': 'L',
    'NEG_ONE': 'R', 'NEG_TWO': 'R',
    'STAR': '∥', 'STAR_2': '∥',
    'UP': 'L', 'DOWN': 'R',
}

# %%
results = []
for name, g in games.items():
    o = outcome(g)
    results.append({
        'game': name,
        'computed': o,
        'expected': expected[name],
        'match': o == expected[name],
    })
df = pd.DataFrame(results)
df.to_csv(output_dir / '00_smoke_outcomes.csv', index=False)
print(df.to_string(index=False))

# %%
witness_up_star = outcome(add(UP, STAR))
witness_one_star = outcome(add(ONE, STAR))
print(f"o(UP + STAR) = {witness_up_star}  (expected: ∥)")
print(f"o(ONE + STAR) = {witness_one_star}  (expected: L)")
print(f"Witness reproduces: {witness_up_star == '∥' and witness_one_star == 'L'}")


# %%
def compute_outcome(g):
    return outcome(g)


outcomes_parallel = parallel_map(
    compute_outcome,
    list(games.values()),
    n_jobs=n_jobs,
    verbose=0,
    backend='threading',
)
print(f"Parallel computed {len(outcomes_parallel)} outcomes successfully.")

# %% [markdown]
# ## Verdict
#

# %%
all_match = df['match'].all()
witness_ok = witness_up_star == '∥' and witness_one_star == 'L'
verdict = "PASS" if (all_match and witness_ok) else "FAIL"
print(f"VERDICT: {verdict}")
assert verdict == "PASS"
