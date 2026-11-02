"""Tests of scripts/valid_smiles.py; skipped when RDKit isn't installed."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from valid_smiles import (  # noqa: E402
    canonical,
    canonicalize_all,
    group_candidates,
    input_formula,
    mol_formula,
    parse,
    score,
)


def score_lists(candidates, targets, workers=1, ignore_spaces=False, formulas=None):
    """formulas defaults to each reference's own, as its input would give it."""
    lines = [c for spectrum in candidates for c in spectrum]
    molecules = canonicalize_all([*lines, *targets], workers)
    if formulas is None:
        formulas = [molecules[t].formula if molecules[t] else None for t in targets]
    return score(candidates, targets, molecules, formulas, ignore_spaces)


def test_canonical_rejects_invalid_smiles():
    assert canonical("") is None
    assert canonical("C1CC") is None  # unclosed ring
    assert canonical("OCC") == canonical("CCO") == "CCO"


def test_canonical_reads_space_separated_tokens():
    # As stored in tgt-*.txt: without joining, RDKit would read only "C".
    assert canonical("C c 1 c c c ( Cl ) c c 1") == canonical("Cc1ccc(Cl)cc1")
    assert canonical("N [C@@H] ( C ) C ( = O ) O") == canonical("N[C@@H](C)C(=O)O")
    assert canonical("C c 1 c c c") is None  # unclosed ring, not just "C"
    assert canonical("c 1 c c c c c 1") is not None


def test_invalid_candidate_no_longer_takes_a_rank():
    result = score_lists([["C1CC", "CCO", "N"]], ["CCO"])
    assert result["top_n_matches"]["raw_exact"] == [0, 1, 1]
    assert result["top_n_matches"]["valid_exact"] == [1, 1, 1]
    assert result["validity"]["valid_at_rank"] == [0.0, 1.0, 1.0]
    assert result["validity"]["top_1_valid"] == 0.0


def test_canonical_counts_the_same_molecule_written_differently():
    result = score_lists([["OCC", "N"]], ["CCO"])
    assert result["top_n_matches"]["valid_exact"] == [0, 0]
    assert result["top_n_matches"]["valid_canonical"] == [1, 1]


def test_canonical_drops_repeated_molecules_before_ranking():
    # OCC and C(O)C are both ethanol, so the reference moves up to rank 2.
    result = score_lists([["OCC", "C(O)C", "CCN"]], ["CCN"])
    assert result["top_n_matches"]["valid_exact"] == [0, 0, 1]
    assert result["top_n_matches"]["valid_canonical"] == [0, 1, 1]


def test_reference_rdkit_cannot_parse_is_counted_not_dropped():
    result = score_lists([["C1CC", "N"], ["C", "N"]], ["C1CC", "C"])
    assert result["samples"] == 2
    assert result["validity"]["invalid_references"] == 1
    assert result["top_n_matches"]["raw_exact"] == [2, 2]
    assert result["top_n_matches"]["valid_canonical"] == [1, 1]
    assert result["top_n_exact_match"]["valid_canonical"]["top_1"] == 0.5


def test_no_space_predictions_against_spaced_references():
    candidates, targets = [["CC(=O)O", "N"]], ["C C ( = O ) O"]
    as_written = score_lists(candidates, targets)
    assert as_written["top_n_matches"]["raw_exact"] == [0, 0]
    assert as_written["top_n_matches"]["valid_canonical"] == [1, 1]
    without_spaces = score_lists(candidates, targets, ignore_spaces=True)
    assert without_spaces["top_n_matches"]["raw_exact"] == [1, 1]
    assert without_spaces["top_n_matches"]["valid_exact"] == [1, 1]


def test_input_formula_reads_the_spaced_formula_the_input_starts_with():
    source = "C 26 H 25 F N 2 O 3 HNMR 7.46 m 1H | 7.42 m 1H |  13CNMR 163.2 161.3"
    assert input_formula(source) == "C26H25FN2O3"
    # A count of 1 is left out, here for O; two-letter elements are one token.
    assert input_formula("C 11 H 14 O S 3 HNMR 7.27 d 8.4Hz 2H") == "C11H14OS3"
    assert input_formula("C 6 H 4 Br Cl 1HNMR 7.3 m 4H") == "C6H4BrCl"
    # The tiny smoke dataset's inputs have none.
    assert input_formula("1HNMR 0.50 1H m 7.26 1H s | 13CNMR 20.0") is None
    assert input_formula("") is None


def test_mol_formula_counts_implicit_hydrogens():
    for smiles, formula in [
        ("C C O", "C 2 H 6 O"),
        ("c1ccccc1", "C 6 H 6"),
        ("Clc1ccc(Br)cc1", "C 6 H 4 Br Cl"),
        ("C[N+](C)(C)C", "C 4 H 12 N"),  # the charge isn't part of the input's formula
        ("[2H]C([2H])([2H])O", "C H 4 O"),  # nor are isotopes
        ("O", "H 2 O"),  # without carbon, all elements alphabetical
    ]:
        assert mol_formula(parse(smiles)) == input_formula(formula + " HNMR"), smiles


def test_formula_filter_drops_candidates_with_another_formula():
    # CCN is C2H7N, the input says C2H6O: the reference moves up to rank 1.
    result = score_lists([["CCN", "OCC", "C"]], ["CCO"])
    assert result["top_n_matches"]["valid_canonical"] == [0, 1, 1]
    assert result["top_n_matches"]["formula_canonical"] == [1, 1, 1]
    assert result["formula"]["right_formula_at_rank"] == [0.0, 1.0, 0.0]
    assert result["formula"]["top_1_right_formula"] == 0.0
    assert result["formula"]["any_right_formula"] == 1.0
    assert result["formula"]["right_formula_candidates"] == pytest.approx(1 / 3)


def test_vote_ranks_the_molecule_written_most_often_first():
    # Dimethyl ether is ranked first, but ethanol is written twice.
    candidates = [["COC", "OCC", "C(O)C", "CCN"]]
    result = score_lists(candidates, ["CCO"])
    assert result["top_n_matches"]["formula_canonical"] == [0, 1, 1, 1]
    assert result["top_n_matches"]["formula_vote"] == [1, 1, 1, 1]


def test_vote_ties_keep_beam_order():
    # One vote each: dimethyl ether stays ahead of ethanol.
    result = score_lists([["COC", "OCC"]], ["CCO"])
    assert result["top_n_matches"]["formula_vote"] == [0, 1]
    # A second vote for ethanol moves it ahead.
    result = score_lists([["COC", "OCC", "CCO"]], ["COC"])
    assert result["top_n_matches"]["formula_vote"] == [0, 1, 1]


def test_input_without_formula_keeps_all_valid_candidates():
    result = score_lists([["CCN", "OCC"], ["CCN", "OCC"]], ["CCO", "CCO"], formulas=[None, "C2H6O"])
    assert result["formula"]["inputs_without_formula"] == 1
    assert result["top_n_matches"]["formula_canonical"] == [1, 2]
    # Shares are over the one spectrum that has a formula.
    assert result["formula"]["right_formula_at_rank"] == [0.0, 1.0]


def test_reference_with_another_formula_than_the_input_is_counted():
    result = score_lists([["CCO"], ["CCO"]], ["CCO", "CCO"], formulas=["C2H6O", "C2H7N"])
    assert result["formula"]["references_wrong_formula"] == 1
    assert result["top_n_matches"]["formula_canonical"] == [1]
    assert result["top_n_matches"]["valid_canonical"] == [2]


def test_group_candidates_infers_outputs_and_rejects_bad_counts():
    assert group_candidates(["a", "b", "c", "d"], 2) == [["a", "b"], ["c", "d"]]
    with pytest.raises(ValueError):
        group_candidates(["a", "b", "c"], 2)
    with pytest.raises(ValueError):
        group_candidates(["a", "b", "c", "d"], 2, num_outputs=3)


def test_workers_give_the_same_result():
    candidates = [["C1CC", "CCO", "OCC"], ["c1ccccc1", "C", "N"], ["X", "Y", "CCN"]]
    targets = ["CCO", "C1=CC=CC=C1", "CCN"]
    assert score_lists(candidates, targets, workers=1) == score_lists(candidates, targets, workers=3)


def test_command_line_writes_json(tmp_path):
    (tmp_path / "tgt-test.txt").write_text("CCO\nCCN\nC\n")
    (tmp_path / "src-test.txt").write_text(
        "C 2 H 6 O HNMR 1.2 t 3H\nC 2 H 7 N HNMR 1.1 t 3H\nC H 4 HNMR 0.2 s 4H\n"
    )
    predictions = tmp_path / "prd-test.txt"
    # Spectra 1 and 2 of 0-2: a chunk starting at index 1.
    predictions.write_text("C1CC\nNCC\nC\nO\n")
    subprocess.run(
        [sys.executable, str(REPO_DIR / "scripts" / "valid_smiles.py"), str(predictions),
         "--data-dir", str(tmp_path), "--start-index", "1", "--workers", "1"],
        check=True, capture_output=True,
    )
    result = json.loads((tmp_path / "prd-test_valid.json").read_text())
    assert result["samples"] == 2
    assert result["num_outputs"] == 2
    assert result["top_n_matches"]["valid_exact"] == [1, 1]
    assert result["top_n_matches"]["valid_canonical"] == [2, 2]
    assert result["top_n_matches"]["formula_vote"] == [2, 2]
    # Of the 4 candidates, NCC (C2H7N) and C (CH4) have their input's formula.
    assert result["formula"]["right_formula_candidates"] == 0.5
    assert result["formula"]["references_wrong_formula"] == 0
    assert result["inputs_file"] == str(tmp_path / "src-test.txt")
    assert result["ignore_spaces"] is False


def test_command_line_detects_spacing_from_the_predictions(tmp_path):
    (tmp_path / "tgt-test.txt").write_text("C C O\nC C N\n")
    (tmp_path / "src-test.txt").write_text("C 2 H 6 O HNMR 1.2 t 3H\nC 2 H 7 N HNMR 1.1 t 3H\n")
    predictions = tmp_path / "prd-test.txt"
    predictions.write_text("CCO\nCCC\n")
    stdout = subprocess.run(
        [sys.executable, str(REPO_DIR / "scripts" / "valid_smiles.py"), str(predictions),
         "--data-dir", str(tmp_path), "--workers", "1"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "model nospace (from the predictions), references spaced: compared without spaces" in stdout
    result = json.loads((tmp_path / "prd-test_valid.json").read_text())
    assert result["model_spacing"] == "nospace" and result["reference_spacing"] == "spaced"
    assert result["top_n_matches"]["raw_exact"] == [1]
