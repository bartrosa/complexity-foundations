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
# Experiment: [TITLE]
# Hypothesis: [what we're testing]
# Kill criterion: [what failure looks like]
# Survive criterion: [what success looks like]
# Source paper / claim: [where this result lives, if anywhere]
#

# %% tags=["parameters"]
# Papermill parameters - override via papermill -p key value
random_seed = 42
output_dir = 'data'
figures_dir = 'figures'
n_jobs = -1

# %%
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import pandas as pd  # noqa: F401

from conway_foundations.utils.parallel import (
    detect_compute_backend,
    parallel_map,  # noqa: F401
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

# %% [markdown]
# ## Verdict
#

# %%
verdict = "PASS"  # or "FAIL" or "PARTIAL"
print(f"VERDICT: {verdict}")
