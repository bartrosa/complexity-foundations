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
# Experiment: GC(φ) topological invariants vs DPLL hardness
#
# Statistical analysis of `gc_phi_data.csv`: Spearman correlations, bootstrap CIs, **partial correlations** (controlling for size proxies), consolidated verdict with **`gc_phi_homology`** scaling summary.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

output_dir = Path("data")
figures_dir = Path("figures")

df = pd.read_csv(output_dir / "gc_phi_data.csv")

invariant_cols = [
    c
    for c in df.columns
    if c.startswith("betti_")
    or c.startswith("n_simplices")
    or c in ("n_cover_relations", "n_full_order")
]

corrs = []
for n in sorted(df["n"].unique()):
    sub = df[df["n"] == n]
    for inv in invariant_cols:
        if sub[inv].nunique() < 2:
            continue
        rho, p = stats.spearmanr(sub[inv], sub["mean_backtracks"])
        corrs.append(
            {
                "n": n,
                "invariant": inv,
                "spearman_r": rho,
                "p_value": p,
                "n_samples": len(sub),
            }
        )
corr_df = pd.DataFrame(corrs)
corr_df.to_csv(output_dir / "gc_dpll_correlations.csv", index=False)
print(corr_df.to_string(index=False))

# %%
def bootstrap_spearman(x, y, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    rhos = []
    n = len(x)
    if n < 4:
        return (np.nan, np.nan)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        x_b = x.iloc[idx]
        y_b = y.iloc[idx]
        if x_b.nunique() > 1 and y_b.nunique() > 1:
            r, _ = stats.spearmanr(x_b, y_b)
            rhos.append(r)
    if not rhos:
        return (np.nan, np.nan)
    return tuple(np.percentile(rhos, [2.5, 97.5]))


bootstrap_data = []
for n in sorted(df["n"].unique()):
    sub = df[df["n"] == n]
    for inv in invariant_cols:
        if sub[inv].nunique() < 2:
            continue
        ci = bootstrap_spearman(sub[inv], sub["mean_backtracks"])
        rho, p = stats.spearmanr(sub[inv], sub["mean_backtracks"])
        bootstrap_data.append(
            {
                "n": n,
                "invariant": inv,
                "spearman_r": rho,
                "ci_low": ci[0],
                "ci_high": ci[1],
                "p_value": p,
                "robust": (
                    not np.isnan(ci[0])
                    and not np.isnan(ci[1])
                    and ci[0] * ci[1] > 0
                    and abs(ci[0]) > 0.2
                    and abs(ci[1]) > 0.2
                ),
            }
        )
bootstrap_df = pd.DataFrame(bootstrap_data)
bootstrap_df.to_csv(output_dir / "gc_dpll_bootstrap.csv", index=False)
print("Bootstrap-robust correlations:")
print(bootstrap_df[bootstrap_df["robust"]].to_string(index=False))

# %%
uniq_n = sorted(df["n"].unique())
fig, axes = plt.subplots(1, len(uniq_n), figsize=(6 * len(uniq_n), 6))
if len(uniq_n) == 1:
    axes = [axes]

for ax, n in zip(axes, uniq_n, strict=False):
    sub = corr_df[corr_df["n"] == n].set_index("invariant")["spearman_r"]
    sub.plot(kind="barh", ax=ax)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.5)
    ax.axvline(-0.3, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Spearman ρ vs DPLL backtracks")
    ax.set_title(f"n = {n}")
    ax.set_xlim(-1, 1)

plt.tight_layout()
plt.savefig(figures_dir / "fig_gc_dpll_correlations.pdf", bbox_inches="tight")
plt.show()

# %%
robust_results = bootstrap_df[bootstrap_df["robust"]]
n_robust_invariants = robust_results["invariant"].nunique()
n_robust_regimes = (
    robust_results.groupby("invariant")["n"].nunique().reset_index(name="n_regimes")
)
robust_with_regime_coverage = n_robust_regimes[n_robust_regimes["n_regimes"] >= 2]

print(f"Robust correlations (CI doesn't cross 0, |ρ|>0.2): {len(robust_results)}")
print(f"Distinct invariants with robust correlation: {n_robust_invariants}")
print(f"Invariants robust across ≥2 regimes: {len(robust_with_regime_coverage)}")
print(robust_with_regime_coverage.to_string(index=False))

if len(robust_with_regime_coverage) >= 1:
    verdict_bootstrap = "PASS"
elif len(robust_results) >= 1:
    verdict_bootstrap = "PARTIAL"
else:
    verdict_bootstrap = "FAIL"
print(f"\nBootstrap verdict: {verdict_bootstrap}")

# %% [markdown]
# ### Partial correlations (control for GC size)
#
# Tests whether Betti numbers carry information beyond simplex counts / order-relation counts.

# %%
try:
    import pingouin as pg

    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    print("pingouin not installed — using manual rank-residualization")

# %%
def manual_partial_correlation(df_local, x_col, y_col, control_cols):
    """Spearman partial correlation via rank residualization (OLS on ranks)."""
    sub = df_local[[x_col, y_col] + control_cols].dropna()
    if len(sub) < len(control_cols) + 3:
        return {"r": np.nan, "p_value": np.nan, "n": len(sub)}

    sub_ranked = sub.rank()
    x_mat = sub_ranked[control_cols].values

    try:
        from sklearn.linear_model import LinearRegression

        reg_x = LinearRegression().fit(x_mat, sub_ranked[x_col].values)
        resid_x = sub_ranked[x_col].values - reg_x.predict(x_mat)

        reg_y = LinearRegression().fit(x_mat, sub_ranked[y_col].values)
        resid_y = sub_ranked[y_col].values - reg_y.predict(x_mat)
    except ImportError:
        ones = np.ones((len(x_mat), 1))
        x_aug = np.hstack([x_mat, ones])
        coef_x, *_ = np.linalg.lstsq(x_aug, sub_ranked[x_col].values, rcond=None)
        resid_x = sub_ranked[x_col].values - x_aug @ coef_x
        coef_y, *_ = np.linalg.lstsq(x_aug, sub_ranked[y_col].values, rcond=None)
        resid_y = sub_ranked[y_col].values - x_aug @ coef_y

    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return {"r": np.nan, "p_value": np.nan, "n": len(sub)}
    r, p = stats.pearsonr(resid_x, resid_y)
    return {"r": float(r), "p_value": float(p), "n": len(sub)}


def partial_corr(df_local, x_col, y_col, control_cols):
    """Partial Spearman via pingouin or manual fallback."""
    if PINGOUIN_AVAILABLE:
        sub = df_local[[x_col, y_col] + control_cols].dropna()
        if len(sub) < len(control_cols) + 3:
            return {"r": np.nan, "p_value": np.nan, "n": len(sub)}
        result = pg.partial_corr(
            data=sub,
            x=x_col,
            y=y_col,
            covar=control_cols,
            method="spearman",
        )
        p_col = next((c for c in ("p-val", "p_val", "pval") if c in result.columns), None)
        if p_col is None:
            raise KeyError(f"partial_corr: no p-value column in {list(result.columns)}")
        return {
            "r": float(result["r"].iloc[0]),
            "p_value": float(result[p_col].iloc[0]),
            "n": int(result["n"].iloc[0]),
        }
    return manual_partial_correlation(df_local, x_col, y_col, control_cols)

# %%
size_invariants = ["n_simplices_total", "n_full_order", "n_cover_relations"]
homology_invariants = [c for c in df.columns if c.startswith("betti_")]

partial_results = []
for n in sorted(df["n"].unique()):
    sub = df[df["n"] == n].copy()
    for inv in homology_invariants:
        if inv not in sub.columns:
            continue
        if sub[inv].nunique() < 2:
            continue
        marg_r, marg_p = stats.spearmanr(sub[inv], sub["mean_backtracks"])
        controls = [c for c in size_invariants if c in sub.columns and sub[c].nunique() > 1]
        if not controls:
            partial_r, partial_p = np.nan, np.nan
            partial_n = len(sub)
        else:
            pc = partial_corr(sub, inv, "mean_backtracks", controls)
            partial_r, partial_p, partial_n = pc["r"], pc["p_value"], pc["n"]
        partial_results.append(
            {
                "n": n,
                "invariant": inv,
                "controls": ",".join(controls),
                "marginal_r": marg_r,
                "marginal_p": marg_p,
                "partial_r": partial_r,
                "partial_p": partial_p,
                "shrinkage": abs(marg_r) - abs(partial_r)
                if not np.isnan(partial_r)
                else np.nan,
                "sample_n": partial_n,
            }
        )

partial_df = pd.DataFrame(partial_results)
partial_df.to_csv(output_dir / "gc_dpll_partial_correlations.csv", index=False)
print("Marginal vs partial correlations (controlling for size invariants):")
print(partial_df.to_string(index=False))

# %%
partial_uniq_n = sorted(df["n"].unique())
fig, axes = plt.subplots(1, len(partial_uniq_n), figsize=(6 * len(partial_uniq_n), 5))
if len(partial_uniq_n) == 1:
    axes = [axes]

for ax, n in zip(axes, partial_uniq_n, strict=False):
    sub = partial_df[partial_df["n"] == n]
    if len(sub) == 0:
        ax.set_title(f"n = {n} (no homology data)")
        continue
    x = np.arange(len(sub))
    width = 0.35
    ax.barh(x - width / 2, sub["marginal_r"], width, label="Marginal ρ", color="steelblue")
    ax.barh(x + width / 2, sub["partial_r"], width, label="Partial ρ (size controlled)", color="coral")
    ax.set_yticks(x)
    ax.set_yticklabels(sub["invariant"])
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.axvline(0.3, color="red", linestyle="--", alpha=0.4)
    ax.axvline(-0.3, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Correlation vs DPLL backtracks")
    ax.set_title(f"n = {n}")
    ax.set_xlim(-1, 1)
    ax.legend()
plt.tight_layout()
plt.savefig(figures_dir / "fig_partial_correlations.pdf", bbox_inches="tight")
plt.show()

# %%
significant_after_partial = partial_df[
    (partial_df["partial_p"] < 0.05) & (partial_df["partial_r"].abs() > 0.2)
]
print(
    "\nHomology invariants with significant partial correlation "
    "(|r|>0.2, p<0.05) after controlling for size:",
)
if len(significant_after_partial) > 0:
    print(
        significant_after_partial[["n", "invariant", "partial_r", "partial_p"]].to_string(
            index=False
        )
    )
else:
    print("None — homological signal may be captured by size invariants.")

mean_shrinkage = partial_df["shrinkage"].mean()
print(f"\nMean shrinkage (|marginal_r| - |partial_r|): {mean_shrinkage:.3f}")
print(f"Max shrinkage: {partial_df['shrinkage'].max():.3f}")

if mean_shrinkage < 0.3 and len(significant_after_partial) > 0:
    partial_scenario = "X — partial correlations survive (homology beyond size)"
elif mean_shrinkage >= 0.3 or len(significant_after_partial) == 0:
    partial_scenario = "Y — high shrinkage or no significant partials (size-driven)"
else:
    partial_scenario = "mixed"

print(f"\nPartial-correlation scenario (documentation): {partial_scenario}")

# %%
scaling_summary_path = output_dir / "gc_phi_scaling_summary.json"
if scaling_summary_path.exists():
    with open(scaling_summary_path, encoding="utf-8") as f:
        scaling_loaded = json.load(f)
    scaling_verdict = scaling_loaded.get("scaling_verdict", "UNKNOWN")
    scales_well_flag = scaling_loaded.get("scales_well", False)
else:
    scaling_loaded = {}
    scaling_verdict = "UNKNOWN — run notebooks/gc_phi_homology.ipynb scaling section first"
    scales_well_flag = False
    print("Note:", scaling_verdict)

n_partial_significant = len(significant_after_partial)
homology_independent = n_partial_significant > 0 and mean_shrinkage < 0.3

marginal_ok = bool((corr_df["spearman_r"].abs() > 0.3).any())
n_strong_marginal = int((corr_df["spearman_r"].abs() > 0.5).sum())

print("=" * 60)
print("ROUND D CONSOLIDATED VERDICT")
print("=" * 60)

print("\n1. Marginal correlations:")
print(f"   - Any invariant with |ρ|>0.3: {marginal_ok}")
print(f"   - Count with |ρ|>0.5: {n_strong_marginal}")

print("\n2. Partial correlations (controlling for size):")
print(f"   - Homology rows with significant partial |r|>0.2, p<0.05: {n_partial_significant}")
print(f"   - Mean shrinkage |marginal| - |partial|: {mean_shrinkage:.3f}")
print(f"   - Independent homological signal (heuristic): {homology_independent}")

print("\n3. Scaling (n=6,7 — from gc_phi_scaling_summary.json):")
print(f"   - Verdict: {scaling_verdict}")
if scaling_loaded:
    print(f"   - scales_well flag: {scales_well_flag}")

print()
if marginal_ok and homology_independent and "PASS" in scaling_verdict:
    final_verdict = "STRONG PASS — independent homology signal, robust scaling"
elif marginal_ok and "PASS" in scaling_verdict:
    final_verdict = (
        "PASS — size-driven correlation scales well; homology not clearly independent"
    )
elif marginal_ok:
    final_verdict = "PARTIAL — correlations at small n; check scaling"
else:
    final_verdict = "FAIL"

print(f"FINAL VERDICT: {final_verdict}")
