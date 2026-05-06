"""Tests for SAT primitives, game complex GC(φ), and DPLL."""

from conway_foundations.topology.game_complex import (
    build_satisfaction_poset,
    gc_phi_summary,
    order_complex_simplices,
)
from conway_foundations.topology.sat import (
    all_assignments,
    is_satisfiable_brute,
    random_3sat,
    satisfied_clauses,
)
from conway_foundations.topology.sat_hardness import dpll


class TestSATPrimitives:
    def test_random_3sat_size(self):
        phi = random_3sat(5, 10, seed=1)
        assert phi.n_vars == 5
        assert len(phi.clauses) == 10

    def test_all_assignments_count(self):
        assert len(all_assignments(3)) == 8
        assert len(all_assignments(5)) == 32

    def test_satisfaction_indices_valid(self):
        phi = random_3sat(4, 5, seed=2)
        for a in all_assignments(4):
            sat = satisfied_clauses(phi, a)
            assert all(0 <= i < 5 for i in sat)

    def test_dpll_matches_brute_force(self):
        for seed in range(5):
            phi = random_3sat(4, 6, seed=seed)
            brute = is_satisfiable_brute(phi)
            dpll_result, _ = dpll(phi)
            assert brute == dpll_result, f"DPLL/brute mismatch at seed {seed}"


class TestGameComplex:
    def test_poset_includes_all_assignments(self):
        phi = random_3sat(3, 3, seed=5)
        poset = build_satisfaction_poset(phi)
        assert len(poset["vertices"]) == 8

    def test_simplex_count_nontrivial(self):
        phi = random_3sat(4, 5, seed=7)
        poset = build_satisfaction_poset(phi)
        simplices = order_complex_simplices(poset, max_dim=3)
        assert len(simplices) >= len(poset["vertices"])

    def test_summary_returns_betti_list(self):
        phi = random_3sat(3, 4, seed=9)
        summary = gc_phi_summary(phi, max_dim=2)
        assert "betti" in summary
        assert len(summary["betti"]) == 3
        assert all(isinstance(b, int) for b in summary["betti"])
