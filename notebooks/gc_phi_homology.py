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
# Experiment: Game Complex GC(φ) and DPLL coupling
#
# **Hypothesis:** Topological invariants of GC(φ) (Betti numbers, simplex counts) correlate with DPLL solving difficulty (backtrack count) for random 3-SAT instances.
#
# **Kill criterion:** No topological invariant has |Spearman ρ| > 0.3 with mean DPLL backtracks, robust across seeds and (*n*, *m*/*n*) regimes.
#
# **Survive criterion:** At least one invariant correlates with |ρ| > 0.3, *p* < 0.01, bootstrap CI not crossing zero, robust across ≥3 seeds and ≥2 (*n*, *m*/*n*) regimes.
#
# Round B and Round C-revised established non-homomorphism of outcome under disjunctive sum and mapped information-theoretic metrics. Round D is independent: SAT instances, GC(φ), and DPLL hardness.

# %% tags=["parameters"]
random_seed = 42
output_dir = "data"
figures_dir = "figures"
n_jobs = -1

n_values = [3, 4, 5]
ratios = [1.0, 2.0, 3.0, 4.27, 5.0]
n_instances_per_config = 20
hardness_orderings = 5
max_dim_homology = 3

scaling_n_values = [6, 7]
scaling_n_instances = 10
scaling_ratios = [2.0, 4.27, 5.0]
scaling_max_dim = 2

# %%
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from conway_foundations.topology.game_complex import gc_phi_summary
from conway_foundations.topology.sat import is_satisfiable_brute, random_3sat
from conway_foundations.topology.sat_hardness import measure_hardness
from conway_foundations.utils.parallel import detect_compute_backend, parallel_map

np.random.seed(random_seed)
output_dir = Path(output_dir)
figures_dir = Path(figures_dir)
output_dir.mkdir(exist_ok=True, parents=True)
figures_dir.mkdir(exist_ok=True, parents=True)

print("Compute backend:", detect_compute_backend())

# %%
configs = [
    {
        "n": n,
        "m": max(1, int(round(n * r))),
        "ratio": r,
        "seed": s,
    }
    for n in n_values
    for r in ratios
    for s in range(n_instances_per_config)
]
print(f"Total instances: {len(configs)}")

# %%
def process_instance(cfg):
    phi = random_3sat(cfg["n"], cfg["m"], cfg["seed"])
    summary = gc_phi_summary(phi, max_dim=max_dim_homology)
    hardness = measure_hardness(
        phi, n_orderings=hardness_orderings, seed=cfg["seed"]
    )
    sat = is_satisfiable_brute(phi) if cfg["n"] <= 5 else None

    record = dict(cfg)
    record["satisfiable"] = sat
    record["n_simplices_total"] = summary["n_simplices_total"]
    record["n_cover_relations"] = summary["n_cover_relations"]
    record["n_full_order"] = summary["n_full_order"]
    for d, count in summary["n_simplices_by_dim"].items():
        record[f"n_simplices_dim_{d}"] = count
    for d, b in enumerate(summary["betti"]):
        record[f"betti_{d}"] = b
    record.update(hardness)
    return record


results = parallel_map(process_instance, configs, n_jobs=n_jobs, verbose=10)
df = pd.DataFrame(results)
df.to_csv(output_dir / "gc_phi_data.csv", index=False)
print(f"Computed for {len(df)} instances")
df.head()

# %%
def marginal_corr_rows(df_in: pd.DataFrame) -> pd.DataFrame:
    """Same marginal Spearman definition as `dpll_correlation.ipynb`."""
    inv_cols = [
        c
        for c in df_in.columns
        if c.startswith("betti_")
        or c.startswith("n_simplices")
        or c in ("n_cover_relations", "n_full_order")
    ]
    rows = []
    for n in sorted(df_in["n"].unique()):
        sub = df_in[df_in["n"] == n]
        for inv in inv_cols:
            if inv not in sub.columns or sub[inv].nunique() < 2:
                continue
            rho, p = stats.spearmanr(sub[inv], sub["mean_backtracks"])
            rows.append(
                {
                    "n": n,
                    "invariant": inv,
                    "spearman_r": rho,
                    "p_value": p,
                    "n_samples": len(sub),
                }
            )
    return pd.DataFrame(rows)


main_corr_df = marginal_corr_rows(df)

# %%
fig, axes = plt.subplots(1, len(n_values), figsize=(6 * len(n_values), 5))
if len(n_values) == 1:
    axes = [axes]
for ax, n in zip(axes, n_values, strict=False):
    sub = df[df["n"] == n]
    grp = (
        sub.groupby("ratio")
        .agg(
            {
                "mean_backtracks": "mean",
                "betti_1": "mean",
                "n_simplices_total": "mean",
            }
        )
        .reset_index()
    )

    def normalize(x):
        m = x.max()
        return x / m if m > 0 else x

    ax.plot(
        grp["ratio"],
        normalize(grp["mean_backtracks"]),
        marker="o",
        label="DPLL hardness (norm)",
    )
    ax.plot(
        grp["ratio"],
        normalize(grp["betti_1"]),
        marker="s",
        label="Betti_1 (norm)",
    )
    ax.plot(
        grp["ratio"],
        normalize(grp["n_simplices_total"]),
        marker="^",
        label="n_simplices (norm)",
    )
    ax.axvline(4.27, color="gray", linestyle="--", alpha=0.5, label="m/n = 4.27")
    ax.set_xlabel("m / n")
    ax.set_ylabel("Normalized")
    ax.set_title(f"n = {n}")
    ax.legend()

plt.tight_layout()
plt.savefig(figures_dir / "fig_gc_phi_phase.pdf", bbox_inches="tight")
plt.show()

# %%
verdict_phase_plot = "computed"

# %% [markdown]
# ### Scaling sanity check (*n* = 6, 7)
#
# Fewer instances per (*n*, ratio), smaller `max_dim`, subset of ratios — tractability on laptop hardware.

# %%
scaling_configs = [
    {
        "n": n,
        "m": max(1, int(round(n * r))),
        "ratio": r,
        "seed": s,
    }
    for n in scaling_n_values
    for r in scaling_ratios
    for s in range(scaling_n_instances)
]
print(f"Scaling instances: {len(scaling_configs)}")

# %%
def process_instance_scaling(cfg):
    phi = random_3sat(cfg["n"], cfg["m"], cfg["seed"])
    summary = gc_phi_summary(phi, max_dim=scaling_max_dim)
    hardness = measure_hardness(
        phi, n_orderings=hardness_orderings, seed=cfg["seed"]
    )
    record = dict(cfg)
    record["n_simplices_total"] = summary["n_simplices_total"]
    record["n_cover_relations"] = summary["n_cover_relations"]
    record["n_full_order"] = summary["n_full_order"]
    for d, count in summary["n_simplices_by_dim"].items():
        record[f"n_simplices_dim_{d}"] = count
    for d, b in enumerate(summary["betti"]):
        record[f"betti_{d}"] = b
    record.update(hardness)
    return record


t_start = time.time()
scaling_results = parallel_map(
    process_instance_scaling, scaling_configs, n_jobs=n_jobs, verbose=10
)
t_elapsed = time.time() - t_start
print(f"Scaling run completed in {t_elapsed:.1f} seconds")

scaling_df = pd.DataFrame(scaling_results)
scaling_df.to_csv(output_dir / "gc_phi_scaling_data.csv", index=False)
print(f"Scaling data: {len(scaling_df)} instances")
scaling_df.head()

# %%
scaling_invariants = [
    c
    for c in scaling_df.columns
    if c.startswith("betti_")
    or c.startswith("n_simplices")
    or c in ("n_cover_relations", "n_full_order")
]

scaling_corrs = []
for n in sorted(scaling_df["n"].unique()):
    sub = scaling_df[scaling_df["n"] == n]
    for inv in scaling_invariants:
        if inv not in sub.columns or sub[inv].nunique() < 2:
            continue
        rho, p = stats.spearmanr(sub[inv], sub["mean_backtracks"])
        scaling_corrs.append(
            {
                "n": n,
                "invariant": inv,
                "spearman_r": rho,
                "p_value": p,
                "n_samples": len(sub),
            }
        )
scaling_corr_df = pd.DataFrame(scaling_corrs)
scaling_corr_df.to_csv(output_dir / "gc_phi_scaling_correlations.csv", index=False)
print("Scaling correlations:")
print(scaling_corr_df.to_string(index=False))

# %%
combined_corr = pd.concat([main_corr_df, scaling_corr_df], ignore_index=True)
mean_abs_rho_per_n = combined_corr.groupby("n")["spearman_r"].apply(lambda s: s.abs().mean())
print("Mean |ρ| across all invariants, by n:")
print(mean_abs_rho_per_n)

scales_well = bool((mean_abs_rho_per_n > 0.3).all()) if len(mean_abs_rho_per_n) else False
trend = (
    float(mean_abs_rho_per_n.iloc[-1] - mean_abs_rho_per_n.iloc[0])
    if len(mean_abs_rho_per_n) >= 2
    else float("nan")
)
print(f"\nMean |ρ| trend (first to last n in combined table): {trend:+.3f}")

if scales_well and trend >= -0.1:
    scaling_verdict = "PASS — correlations stable or strengthening with n"
elif scales_well:
    scaling_verdict = "PARTIAL — correlations exist at all scales but weakening with n"
else:
    scaling_verdict = "FAIL — correlations weaken substantially at higher n"
print(f"\nScaling verdict: {scaling_verdict}")

scaling_summary = {
    "scaling_verdict": scaling_verdict,
    "mean_abs_rho_by_n": {str(k): float(v) for k, v in mean_abs_rho_per_n.items()},
    "scales_well": scales_well,
    "mean_abs_rho_trend_first_last": trend,
    "scaling_runtime_seconds": t_elapsed,
}
with open(output_dir / "gc_phi_scaling_summary.json", "w", encoding="utf-8") as f:
    json.dump(scaling_summary, f, indent=2)
print("Wrote", output_dir / "gc_phi_scaling_summary.json")

# %%
common_inv = sorted(
    set(main_corr_df["invariant"]) & set(scaling_corr_df["invariant"])
)
all_n = sorted(set(main_corr_df["n"]) | set(scaling_corr_df["n"]))
fig, ax = plt.subplots(figsize=(12, 6))
if not common_inv:
    ax.text(0.5, 0.5, "No common invariants for main vs scaling", ha="center")
else:
    x = np.arange(len(common_inv))
    width = min(0.12, 0.8 / max(len(all_n), 1))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_n)))

    for i, n in enumerate(all_n):
        rhos = []
        for inv in common_inv:
            sub_m = main_corr_df[
                (main_corr_df["n"] == n) & (main_corr_df["invariant"] == inv)
            ]
            sub_s = scaling_corr_df[
                (scaling_corr_df["n"] == n) & (scaling_corr_df["invariant"] == inv)
            ]
            if len(sub_m):
                rhos.append(float(sub_m["spearman_r"].iloc[0]))
            elif len(sub_s):
                rhos.append(float(sub_s["spearman_r"].iloc[0]))
            else:
                rhos.append(float("nan"))
        ax.bar(
            x + (i - (len(all_n) - 1) / 2) * width,
            rhos,
            width,
            label=f"n={n}",
            color=colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(common_inv, rotation=45, ha="right")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.4)
    ax.axhline(-0.3, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("Spearman ρ vs DPLL backtracks")
    ax.set_title("Topological invariant ↔ DPLL hardness — scaling across n")
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right", ncol=min(len(all_n), 4))
plt.tight_layout()
plt.savefig(figures_dir / "fig_scaling_correlations.pdf", bbox_inches="tight")
plt.show()
