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
