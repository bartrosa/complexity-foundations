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
