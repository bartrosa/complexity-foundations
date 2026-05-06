# Experiments index

Status: PASS | FAIL | PARTIAL | NOT-RUN

| Notebook | Hypothesis | Status | Verdict |
|----------|------------|--------|---------|
| 00_smoke | infrastructure works | NOT-RUN | - |
| witness_basics | outcome not an F-algebra morphism under disjunctive sum | PASS | PASS — 12 unique witness pairs, valid φ = 0; outputs include [`witnesses.csv`](../data/witnesses.csv), [`witness_patterns.csv`](../data/witness_patterns.csv), [`witness_class_structure.csv`](../data/witness_class_structure.csv), [`witness_pair_depths.csv`](../data/witness_pair_depths.csv) |
| synergy_library | H1 + H2 (synergy structurally located in ∥-class, correlates with fuzziness) | PASS | FAIL — H1 not supported (Mann–Whitney one-sided ∥ vs L/R *p* = 0.188; median ratio 1.47× \< 2×); H2 not supported (Spearman synergy vs `avg_fuzzy_count`: \|*r*\| ≈ 0.11, *p* ≈ 0.41); notebook verdict FAIL; outputs [`synergy_library.csv`](../data/synergy_library.csv), [`correlations.csv`](../data/correlations.csv) |
| robustness_priors | H3 (PID methods agree on synergy ranking) | PASS | GRACEFUL_FAIL — `dit` does not import under NumPy 2.x (`np.alltrue` removed); BROJA/PPID fall back to Williams–Beer so rank correlations are trivially 1.0; not an independent cross-method test; output [`method_robustness.csv`](../data/method_robustness.csv) |
