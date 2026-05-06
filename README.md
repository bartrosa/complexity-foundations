# complexity_foundations

Computational verification for Conway Machine and CGT-grounded
complexity research.

## Quickstart

```bash
pip install -e ".[all]"
make verify
```

## Layout

- `src/conway_foundations/` — algorithms (testable, type-checked)
- `notebooks/` — experiments (narrative + figures)
- `tests/` — pytest unit tests
- `lean/` — Lean 4 formal claims (later rounds)
- `data/` — experiment outputs (gitignored)
- `figures/` — paper figures (gitignored)
- `docs/` — paper claim status, experiment log, decisions

## Principles

1. Algorithms in `src/`, narrative in notebooks.
2. Every paper number traceable to a notebook.
3. `make verify` reproducible from clean clone.
4. lru_cache for algorithm-level memoization, joblib for experiment-level
   parallelism. No premature GPU optimization.

## Status

| Round | Hypothesis | Verdict | Papers affected |
|-------|------------|---------|-----------------|
| A | bootstrap | pending | infrastructure |
| B | outcome not F-algebra morphism under `+` | PASS | witness search + φ obstruction |
| C | PID synergy systematic (H1 ∥ vs L/R, H2 fuzzy, H3 method agreement) | FAIL / H3 GRACEFUL_FAIL | Mediano letter — no quantitative PID claims without stronger evidence + working `dit` |
| C-revised | Multi-metric synergy exploration + robustness (canonical `add`, CE fix, full-trajectory TE/Markov) | pending full notebook re-run | Run `papermill notebooks/synergy_library.ipynb …` then `robustness_priors`; see `docs/decision_log.md` |
