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
# # Cross-metric and cross-trajectory robustness
#
# **Question:** Are structural correlations from `synergy_library.ipynb` robust to trajectory seed, length *T*, and metric subset?
#
# We compare Markov compound rate (trajectory-dependent) vs KL divergence (independent perturbations) across configurations *T* ∈ {10, 20, 40}, seeds {1, 2, 3}.
#
# **Runtime:** `len(configurations) × len(pairs)` tasks (default **5 × 171 = 855**). Joblib will keep CPU busy for a long time — use **`pair_limit`** for smoke runs, lower **`n_jobs`**, and see joblib progress via **`verbose`**.
#

# %% tags=["parameters"]
import os

random_seed = 42
output_dir = "data"
figures_dir = "figures"
n_jobs = max(1, min(8, (os.cpu_count() or 4) // 2))
# None = all unordered pairs (171). Set e.g. 40 for a faster partial grid.
pair_limit = None
parallel_verbose = 5


# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from conway_foundations.games.arithmetic import clear_arithmetic_cache
from conway_foundations.games.library import CANONICAL_LIBRARY_EXPANDED
from conway_foundations.games.outcome import clear_outcome_cache
from conway_foundations.games.properties import all_properties
from conway_foundations.synergy import METRIC_REGISTRY
from conway_foundations.synergy.trajectories import generate_trajectory
from conway_foundations.utils.parallel import parallel_map

np.random.seed(random_seed)
output_dir = Path(output_dir)
figures_dir = Path(figures_dir)
output_dir.mkdir(exist_ok=True, parents=True)
figures_dir.mkdir(exist_ok=True, parents=True)


# %%
library = CANONICAL_LIBRARY_EXPANDED
pairs = []
for i, (n1, g1) in enumerate(library):
    for j, (n2, g2) in enumerate(library):
        if i < j:
            pairs.append((n1, n2, g1, g2))
if pair_limit is not None:
    pairs = pairs[: int(pair_limit)]
print(f"Pairs in robustness grid: {len(pairs)} (pair_limit={pair_limit!r})")


# %%
configurations = [
    {"T": 10, "seed": 1},
    {"T": 20, "seed": 1},
    {"T": 20, "seed": 2},
    {"T": 20, "seed": 3},
    {"T": 40, "seed": 1},
]


def compute_subset_for_config(args):
    cfg, pair = args
    n1, n2, g1, g2 = pair
    traj = generate_trajectory(library, length=cfg["T"], seed=cfg["seed"])
    indep = [g for _, g in library]

    markov = METRIC_REGISTRY["markov_entropy_rate"]
    kl = METRIC_REGISTRY["kl_divergence"]

    clear_outcome_cache()
    clear_arithmetic_cache()
    m_res = markov(g1, g2, traj)
    clear_outcome_cache()
    clear_arithmetic_cache()
    k_res = kl(g1, g2, indep)

    p1 = all_properties(g1)
    p2 = all_properties(g2)

    return {
        "config_T": cfg["T"],
        "config_seed": cfg["seed"],
        "pair": f"{n1}+{n2}",
        "avg_canonical_size": (int(p1["canonical_size"]) + int(p2["canonical_size"])) / 2,
        "avg_depth": (int(p1["depth"]) + int(p2["depth"])) / 2,
        "markov_compound": m_res["markov_rate_compound"],
        "kl": k_res["kl_joint_vs_indep"],
    }


config_pair_combos = [(cfg, pair) for cfg in configurations for pair in pairs]
print(
    f"parallel_map tasks: {len(config_pair_combos)} "
    f"({len(configurations)} configs × {len(pairs)} pairs), n_jobs={n_jobs}",
)
robust_results = parallel_map(
    compute_subset_for_config,
    config_pair_combos,
    n_jobs=n_jobs,
    verbose=parallel_verbose,
)
robust_df = pd.DataFrame(robust_results)
robust_df.to_csv(output_dir / "cross_config_metrics.csv", index=False)


# %%
stability = []
for cfg_id, group in robust_df.groupby(["config_T", "config_seed"]):
    for metric in ["markov_compound", "kl"]:
        for prop in ["avg_canonical_size", "avg_depth"]:
            rho, p = stats.spearmanr(group[prop], group[metric], nan_policy="omit")
            if np.isnan(rho):
                continue
            stability.append(
                {
                    "T": cfg_id[0],
                    "seed": cfg_id[1],
                    "metric": metric,
                    "property": prop,
                    "rho": float(rho),
                    "p": float(p),
                }
            )
stab_df = pd.DataFrame(stability)
stab_df.to_csv(output_dir / "config_stability.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric in zip(axes, ["markov_compound", "kl"], strict=True):
    sub = stab_df[stab_df["metric"] == metric]
    for prop in ["avg_canonical_size", "avg_depth"]:
        sub_p = sub[sub["property"] == prop].sort_values(["T", "seed"])
        labels = [f"T={int(r['T'])}, s={int(r['seed'])}" for _, r in sub_p.iterrows()]
        ax.plot(labels, sub_p["rho"].values, marker="o", label=prop)
    ax.set_title(metric)
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(figures_dir / "fig_config_stability.pdf", bbox_inches="tight")
plt.show()


# %%
def correlation_robust(rho: pd.Series) -> bool:
    if rho.empty:
        return False
    if abs(rho.mean()) <= 0.3:
        return False
    return bool(((rho > 0).all()) | ((rho < 0).all()))


stable_correlations = stab_df.groupby(["metric", "property"])["rho"].agg(["mean", "std", "min", "max"])
print("Stability across trajectory configurations:")
print(stable_correlations)

robust_flags = (
    stab_df.groupby(["metric", "property"])["rho"].apply(correlation_robust).rename("robust")
)
print("\nRobust (signed) correlations:")
print(robust_flags)

verdict = "PASS" if robust_flags.any() else "FAIL"
print(f"\nVERDICT: {verdict}")
