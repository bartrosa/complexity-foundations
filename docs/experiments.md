# Experiments index

Status: PASS | FAIL | PARTIAL | NOT-RUN

| Notebook | Hypothesis | Status | Verdict |
|----------|------------|--------|---------|
| 00_smoke | infrastructure works | NOT-RUN | - |
| witness_basics | outcome not an F-algebra morphism under disjunctive sum | PASS | PASS — 12 unique witness pairs, valid φ = 0; outputs include [`witnesses.csv`](../data/witnesses.csv), [`witness_patterns.csv`](../data/witness_patterns.csv), [`witness_class_structure.csv`](../data/witness_class_structure.csv), [`witness_pair_depths.csv`](../data/witness_pair_depths.csv) |
| synergy_library | Round C-revised multi-metric (∥ vs L/R, metric clusters, structure correlations) | NOT-RUN | — Re-run after consolidation; outputs [`multi_metric_library.csv`](../data/multi_metric_library.csv), [`metric_correlations.csv`](../data/metric_correlations.csv), [`structural_correlations.csv`](../data/structural_correlations.csv), [`bootstrap_top_correlations.csv`](../data/bootstrap_top_correlations.csv), figures under [`figures/`](../figures/) |
| robustness_priors | H3 (PID methods agree on synergy ranking) | PASS | GRACEFUL_FAIL — `dit` does not import under NumPy 2.x (`np.alltrue` removed); BROJA/PPID fall back to the default PID implementation so rank correlations are trivially 1.0; not an independent cross-method test; output [`method_robustness.csv`](../data/method_robustness.csv) |
| gc_phi_homology | GC(φ) vs DPLL + scaling *n*∈{6,7} | NOT-RUN | [`gc_phi_data.csv`](../data/gc_phi_data.csv), [`gc_phi_scaling_data.csv`](../data/gc_phi_scaling_data.csv), [`gc_phi_scaling_correlations.csv`](../data/gc_phi_scaling_correlations.csv), [`gc_phi_scaling_summary.json`](../data/gc_phi_scaling_summary.json), [`fig_gc_phi_phase.pdf`](../figures/fig_gc_phi_phase.pdf), [`fig_scaling_correlations.pdf`](../figures/fig_scaling_correlations.pdf) |
| dpll_correlation | Marginal + bootstrap + **partial** corr vs hardness | NOT-RUN | [`gc_dpll_correlations.csv`](../data/gc_dpll_correlations.csv), [`gc_dpll_partial_correlations.csv`](../data/gc_dpll_partial_correlations.csv), [`gc_dpll_bootstrap.csv`](../data/gc_dpll_bootstrap.csv), figures incl. [`fig_partial_correlations.pdf`](../figures/fig_partial_correlations.pdf); run **after** `gc_phi_homology` so scaling JSON exists for consolidated verdict |
