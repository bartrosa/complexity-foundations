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
# # Systematic non-homomorphism witnesses
#
# Hypothesis: outcome is not an F-algebra morphism under disjunctive sum.
#
# Kill criterion: Some φ: 4×4→4 correctly predicts all compound outcomes in library, OR fewer than 5 witness pairs found.
#
# Survive criterion: ≥5 distinct witnesses AND no φ works.
#
# Source paper / claim: Mediano letter contrast (↑+* vs 1+*); structural obstruction to outcome-only fusion rule.
#

# %% tags=["parameters"]
# Papermill parameters - override via papermill -p key value
random_seed = 42
output_dir = 'data'
figures_dir = 'figures'
n_jobs = -1


# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from conway_foundations.utils.parallel import detect_compute_backend

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
from conway_foundations.coalgebra.morphisms import check_outcome_is_morphism
from conway_foundations.compositionality.witness import find_witnesses, witnesses_to_dataframe
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
    switch,
)
from conway_foundations.games.outcome import outcome

library = [
    ('ZERO', ZERO), ('ONE', ONE), ('TWO', TWO),
    ('NEG_ONE', NEG_ONE), ('NEG_TWO', NEG_TWO),
    ('STAR', STAR), ('STAR_2', STAR_2),
    ('UP', UP), ('DOWN', DOWN),
    ('SWITCH_1', switch(1)), ('SWITCH_2', switch(2)),
]
perturbations = library


# %%
mediano_o_up_star = outcome(add(UP, STAR))
mediano_o_one_star = outcome(add(ONE, STAR))
print(f"o(UP) = {outcome(UP)}, o(ONE) = {outcome(ONE)}  -> both 'L'")
print(f"o(UP + STAR) = {mediano_o_up_star}  (expected: ∥)")
print(f"o(ONE + STAR) = {mediano_o_one_star}  (expected: L)")
mediano_witness_holds = (
    outcome(UP) == outcome(ONE) == 'L'
    and mediano_o_up_star == '∥'
    and mediano_o_one_star == 'L'
)
print(f"Mediano witness: {mediano_witness_holds}")


# %%
witnesses = find_witnesses(library, perturbations)
df = witnesses_to_dataframe(witnesses)
df.to_csv(output_dir / 'witnesses.csv', index=False)
print(f"Found {len(witnesses)} witness instances ({df[['g1_repr','g2_repr']].drop_duplicates().shape[0]} unique pairs)")
print(df.head(10).to_string())


# %%
divergence_patterns = df.groupby(
    ['base_outcome', 'compound_outcome_1', 'compound_outcome_2']
).size().reset_index(name='count')
divergence_patterns.to_csv(output_dir / 'witness_patterns.csv', index=False)
print(divergence_patterns.to_string())


# %%
# Outcome class structure of witnesses
# Question: which (base_outcome, divergence_pattern) classes dominate?

structure_df = df.copy()
structure_df['compound_pair'] = structure_df.apply(
    lambda r: tuple(sorted([r['compound_outcome_1'], r['compound_outcome_2']])),
    axis=1,
)

class_summary = structure_df.groupby(
    ['base_outcome', 'compound_pair']
).size().reset_index(name='witness_count')

class_summary.to_csv(output_dir / 'witness_class_structure.csv', index=False)
print("Witness class structure:")
print(class_summary.to_string(index=False))

# Test hypothesis: does ∥ dominate compound divergences?
parallel_in_compound = structure_df[
    (structure_df['compound_outcome_1'] == '∥')
    | (structure_df['compound_outcome_2'] == '∥')
]
_n = len(df)
_pct = (100 * len(parallel_in_compound) / _n) if _n else 0.0
print(
    f"\nWitnesses with ∥ in compound: {len(parallel_in_compound)}/{_n} ({_pct:.1f}%)"
)


# %%
fig, ax = plt.subplots(figsize=(10, 6))
patterns = divergence_patterns.copy()
patterns['label'] = patterns.apply(
    lambda r: f"{r['base_outcome']} → {r['compound_outcome_1']} vs {r['compound_outcome_2']}",
    axis=1,
)
ax.barh(patterns['label'], patterns['count'])
ax.set_xlabel('Number of witness instances')
ax.set_title('Non-homomorphism witnesses by divergence pattern')
plt.tight_layout()
plt.savefig(figures_dir / 'fig_witnesses.pdf', bbox_inches='tight')
plt.show()


# %%
games_only = [g for _, g in library]
print(f"Library size: {len(games_only)}, pairs: {len(games_only)**2}")
result = check_outcome_is_morphism(games_only, n_jobs=n_jobs)
print(f"Tested {result.tested_phi_count} candidate φ functions")
print(f"Valid φ count: {result.valid_phi_count}")
print(f"Is outcome an F-algebra morphism? {result.is_morphism}")


# %%
# For each unique witness pair, find minimum-depth perturbation that triggers divergence
from conway_foundations.games.properties import depth as game_depth

pair_depth_df = df.groupby(['g1_repr', 'g2_repr']).agg({
    'perturbation_repr': lambda x: list(x.unique()),
}).reset_index()

pert_depth_map = {name: game_depth(g) for name, g in library}

pair_depth_df['min_pert_depth'] = pair_depth_df['perturbation_repr'].apply(
    lambda perts: min(pert_depth_map[p] for p in perts)
)
pair_depth_df['n_pert_witnesses'] = pair_depth_df['perturbation_repr'].apply(len)

pair_depth_df.to_csv(output_dir / 'witness_pair_depths.csv', index=False)
print("\nWitness pairs by min perturbation depth:")
print(pair_depth_df.to_string(index=False))

print(f"\nMean min perturbation depth: {pair_depth_df['min_pert_depth'].mean():.2f}")
print(
    "Pairs needing depth-1 perturbation: "
    f"{(pair_depth_df['min_pert_depth'] == 1).sum()}/{len(pair_depth_df)}"
)


# %% [markdown]
# ## Verdict
#

# %%
unique_pairs = df[['g1_repr', 'g2_repr']].drop_duplicates().shape[0]
witnesses_found = unique_pairs
no_phi_works = not result.is_morphism

if witnesses_found >= 5 and no_phi_works:
    verdict = "PASS"
elif witnesses_found >= 1 and no_phi_works:
    verdict = "PARTIAL"
else:
    verdict = "FAIL"
print(f"Witnesses found: {witnesses_found} unique pairs")
print(f"No valid φ: {no_phi_works}")
print(f"VERDICT: {verdict}")

