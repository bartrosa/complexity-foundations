"""Game complex GC(φ): order complex of the satisfaction-inclusion poset.

Vertices are boolean assignments. Order: a ≤_φ b iff satisfied_clauses(a) ⊆ satisfied_clauses(b).
Chains in this poset (strict inclusion) yield simplices in the order complex.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from conway_foundations.topology.sat import SATInstance, all_assignments, satisfied_clauses


def build_satisfaction_poset(phi: SATInstance) -> dict:
    """Build the partial order on assignments by satisfaction-set inclusion.

    Returns dict with ``vertices``, ``full_order`` pairs (a,b) with a ≤ b,
    ``cover_relations`` (Hasse cover pairs), and ``sat_map``.
    """
    vertices = all_assignments(phi.n_vars)
    sat_map: dict[tuple[bool, ...], frozenset[int]] = {
        a: satisfied_clauses(phi, a) for a in vertices
    }
    full_order: set[tuple[tuple[bool, ...], tuple[bool, ...]]] = set()
    for a in vertices:
        for b in vertices:
            if sat_map[a].issubset(sat_map[b]):
                full_order.add((a, b))
    cover_relations: set[tuple[tuple[bool, ...], tuple[bool, ...]]] = set()
    for a in vertices:
        for b in vertices:
            if a == b:
                continue
            if not sat_map[a].issubset(sat_map[b]) or sat_map[a] == sat_map[b]:
                continue
            between = False
            for c in vertices:
                if c in (a, b):
                    continue
                if (
                    sat_map[a].issubset(sat_map[c])
                    and sat_map[c].issubset(sat_map[b])
                    and sat_map[a] != sat_map[c]
                    and sat_map[c] != sat_map[b]
                ):
                    between = True
                    break
            if not between:
                cover_relations.add((a, b))
    return {
        "vertices": vertices,
        "full_order": full_order,
        "cover_relations": cover_relations,
        "sat_map": sat_map,
    }


def _strict_le(
    sat_map: dict[tuple[bool, ...], frozenset[int]],
    a: tuple[bool, ...],
    b: tuple[bool, ...],
) -> bool:
    return sat_map[a].issubset(sat_map[b]) and sat_map[a] != sat_map[b]


def order_complex_simplices(poset: dict, max_dim: int) -> list[tuple[tuple[bool, ...], ...]]:
    """Simplices = strictly increasing chains in the poset (dimension = chain length − 1)."""
    vertices: list[tuple[bool, ...]] = poset["vertices"]
    sat_map: dict[tuple[bool, ...], frozenset[int]] = poset["sat_map"]
    seen: set[tuple[tuple[bool, ...], ...]] = set()

    def record(chain: tuple[tuple[bool, ...], ...]) -> None:
        if len(chain) == 0:
            return
        if len(chain) - 1 <= max_dim:
            seen.add(chain)

    for v in vertices:
        record((v,))

    def extend(chain: tuple[tuple[bool, ...], ...]) -> None:
        if len(chain) - 1 >= max_dim:
            return
        last = chain[-1]
        for nxt in vertices:
            if _strict_le(sat_map, last, nxt):
                nc = chain + (nxt,)
                record(nc)
                extend(nc)

    for v in vertices:
        extend((v,))

    return list(seen)


def _simplex_to_ints(
    simplex: tuple[tuple[bool, ...], ...],
    v_index: dict[tuple[bool, ...], int],
) -> tuple[int, ...]:
    return tuple(v_index[v] for v in simplex)


def _gf2_rank(mat: np.ndarray) -> int:
    """Rank of a binary matrix over F2."""
    if mat.size == 0:
        return 0
    work = mat.astype(np.int64).copy()
    rows, cols = work.shape
    r = 0
    for c in range(cols):
        pivot = None
        for rr in range(r, rows):
            if work[rr, c] % 2 != 0:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            work[[r, pivot], :] = work[[pivot, r], :]
        for rr in range(rows):
            if rr != r and work[rr, c] % 2 != 0:
                work[rr] = (work[rr] + work[r]) % 2
        r += 1
        if r >= rows:
            break
    return r


def _boundary_matrix_k(
    k: int,
    by_dim: dict[int, list[tuple[int, ...]]],
    idx: dict[int, dict[tuple[int, ...], int]],
) -> np.ndarray:
    """Boundary ∂_k : C_k → C_{k-1} over F2 (rows (k−1)-simplices, cols k-simplices)."""
    if k <= 0:
        return np.zeros((0, 0), dtype=np.int64)
    cols = by_dim.get(k, [])
    rows_idx = idx.get(k - 1, {})
    if not cols or not rows_idx:
        return np.zeros((len(rows_idx), len(cols)), dtype=np.int64)
    boundary = np.zeros((len(rows_idx), len(cols)), dtype=np.int64)
    for j, s in enumerate(cols):
        for i_skip in range(len(s)):
            face = tuple(s[:i_skip] + s[i_skip + 1 :])
            if face in rows_idx:
                row_i = rows_idx[face]
                boundary[row_i, j] = (boundary[row_i, j] + 1) % 2
    return boundary


def _betti_via_chain_complex_int(simplices_int: list[tuple[int, ...]], max_dim: int) -> list[int]:
    """Betti numbers over F2: β_k = n_k − rank(∂_k) − rank(∂_{k+1}) with ∂_0 = 0."""
    by_dim: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for s in simplices_int:
        by_dim[len(s) - 1].append(tuple(s))
    for d in list(by_dim.keys()):
        by_dim[d] = sorted(set(by_dim[d]))

    idx: dict[int, dict[tuple[int, ...], int]] = {}
    for d in range(max_dim + 2):
        lst = by_dim.get(d, [])
        idx[d] = {s: i for i, s in enumerate(lst)}

    betti: list[int] = []
    for k in range(max_dim + 1):
        nk = len(by_dim.get(k, []))
        rk = 0
        if k >= 1:
            dk = _boundary_matrix_k(k, by_dim, idx)
            rk = _gf2_rank(dk) if dk.size else 0
        dkp1 = _boundary_matrix_k(k + 1, by_dim, idx)
        rkp1 = _gf2_rank(dkp1) if dkp1.size else 0
        beta = nk - rk - rkp1
        betti.append(max(0, int(beta)))
    return betti


def betti_numbers_via_gudhi(
    simplices_int: list[tuple[int, ...]],
    max_dim: int,
) -> list[int]:
    """Compute Betti numbers using GUDHI when installed; else chain complex over F2."""
    try:
        import gudhi

        st = gudhi.SimplexTree()
        for simplex in simplices_int:
            if len(simplex) == 0:
                continue
            st.insert(list(simplex), filtration=0.0)
        st.compute_persistence()
        raw = st.betti_numbers()
        return [int(raw[i]) if i < len(raw) else 0 for i in range(max_dim + 1)]
    except ImportError:
        return _betti_via_chain_complex_int(simplices_int, max_dim)
    except Exception:
        return _betti_via_chain_complex_int(simplices_int, max_dim)


def gc_phi_summary(phi: SATInstance, max_dim: int = 3) -> dict:
    """Structural summary of GC(φ): counts and Betti numbers up to ``max_dim``."""
    poset = build_satisfaction_poset(phi)
    simplices = order_complex_simplices(poset, max_dim=max_dim)
    vertices = poset["vertices"]
    v_index = {v: i for i, v in enumerate(vertices)}
    simplices_int = [_simplex_to_ints(s, v_index) for s in simplices]
    betti = betti_numbers_via_gudhi(simplices_int, max_dim=max_dim)
    n_by_dim: dict[int, int] = defaultdict(int)
    for s in simplices:
        d = len(s) - 1
        if d <= max_dim:
            n_by_dim[d] += 1
    return {
        "n_vertices": len(vertices),
        "n_cover_relations": len(poset["cover_relations"]),
        "n_full_order": len(poset["full_order"]),
        "n_simplices_total": len(simplices),
        "n_simplices_by_dim": {d: n_by_dim[d] for d in range(max_dim + 1)},
        "betti": betti,
    }
