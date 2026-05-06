"""Test non-homomorphism witnesses."""

from conway_foundations.coalgebra.morphisms import check_outcome_is_morphism
from conway_foundations.compositionality.witness import find_witnesses
from conway_foundations.games.arithmetic import add
from conway_foundations.games.library import NEG_ONE, ONE, STAR, UP, ZERO
from conway_foundations.games.outcome import outcome


def test_canonical_nonhomomorphism_witness():
    assert outcome(UP) == "L"
    assert outcome(ONE) == "L"
    assert outcome(add(UP, STAR)) == "∥"
    assert outcome(add(ONE, STAR)) == "L"


def test_witnesses_exist_in_minimal_library():
    library = [("ONE", ONE), ("UP", UP), ("STAR", STAR)]
    witnesses = find_witnesses(library, library)
    assert len(witnesses) >= 1
    found_canonical_witness = any(
        {w.g1_repr, w.g2_repr} == {"UP", "ONE"} and w.perturbation_repr == "STAR" for w in witnesses
    )
    assert found_canonical_witness


def test_outcome_not_morphism():
    games = [ZERO, ONE, NEG_ONE, STAR, UP]
    result = check_outcome_is_morphism(games, n_jobs=1)
    assert not result.is_morphism
    assert result.valid_phi_count == 0
