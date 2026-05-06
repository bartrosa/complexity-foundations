# Decision log

Append-only record. Each round documents what survived, what died, why.

## Round A: bootstrap

- No scientific claims yet.
- Infrastructure: notebooks runnable headless, tests pass, lint clean.
- Status: pending verification.

## Round B: non-homomorphism witnesses

- Tested: systematic witness search across canonical library (+ switches); exhaustive φ search with commutative + zero-identity reductions (4096 candidates).
- Verdict: **PASS** — ≥5 distinct base pairs `(g1, g2)` with matching outcomes but divergent compounds under some perturbation; **no** φ: 4×4→4 agrees with all library compound outcomes (valid φ count 0). Mediano contrast (↑+* vs 1+*) reproduced.
- If FAIL: Mediano letter framework broken — alert before any further rounds.
- If PARTIAL: Mediano letter must reframe.
- If PASS: paper A and Mediano letter foundation confirmed (within finite-library obstruction checks).

## Round C-revised: cleanup pass (consolidation, not new experiment)

### What was fixed

1. **Canonical form reduction** — Implemented in `games/canonical.py` (recursive `≤`, domination, reversibility to fixpoint). Integrated into `games/arithmetic.py` so `add` / `neg` return canonical forms; **`add_raw`** remains for internal recursion.
2. **Surreal number fast path** — `games/surreal.py` recognizes dyadic rationals; **`add`** uses **`add_as_numbers`** when both summands are numbers (avoids blowing up integer chains).
3. **Causal emergence** — Replaced incorrect `EI(inputs, inputs)` baseline with **EI_macro − max(EI_micro_g1, EI_micro_g2)** (`metrics_emergence.py`).
4. **`num_distinct_compounds`** — Documented as diagnostic; excluded from primary correlation/clustering in **`synergy_library`** via **`PRIMARY_METRIC_COLS`**.
5. **Trajectory metrics** — Transfer entropy and Markov entropy rate use the **full** sampled trajectory (removed fixed 8/5-step clamps); **`trajectory_length`** parameter defaults to **20**.

### Performance sanity check

- Twenty cumulative **`STAR`** sums + **`outcome`** complete in **≪ 1 s** after canonicalization (see inline benchmark in verification).

### Re-run results (populate locally after full `papermill`)

- Cluster structure: *[after full run]*
- Top metric × structural property correlations (bootstrap CI): *[after full run]*
- Class separation by metric: *[after full run]*
- Causal emergence variance: *[after full run]*
- Verdict: *[after full run]*

### Implications for Mediano letter

- Quantitative claims that survive bootstrap: *[after full run]*
- Recommended framing: *[TBD]*

---

## Round C: PID synergy systematic

- Refined hypotheses based on Round B finding (most witnesses are ∥-class).
- **H1 verdict (∥-pairs have higher synergy):** FAIL — median Williams–Beer synergy 0.2120 bits (∥-base pairs) vs 0.1441 bits (L/R-base pairs); ratio 1.47× (hypothesis required \>2×); Mann–Whitney one-sided *p* = 0.188 (kill threshold: distinguishability requires *p* \< 0.1 for “not distinguishable” failure mode).
- **H2 verdict (synergy correlates with fuzzy_count):** FAIL — Spearman *ρ*(synergy, `avg_fuzzy_count`) ≈ 0.11 (*p* ≈ 0.41) for WB/BROJA/PPID (latter two identical to WB here).
- **H3 verdict (methods agree):** GRACEFUL_FAIL — `dit` unavailable in this environment (import fails on NumPy 2.x); BROJA and PPID collapse to Williams–Beer; rank correlations are not informative.
- **Composite verdict:** FAIL on primary quantitative claims (H1 + H2). H3 is not testable as stated without a working `dit`/compatible NumPy stack.
- **Implication for Mediano letter:** Reframe as **non-homomorphism without quantitative PID claims** unless future work pins synergy structure with a repaired toolchain and stronger effects on this library; any PID wording should be **Williams–Beer–specific** and clearly labeled exploratory if retained.
