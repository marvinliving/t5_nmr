"""Tests of scripts/valid_smiles.py; skipped when RDKit isn't installed."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))

from valid_smiles import canonical, canonicalize_all, group_candidates, score  # noqa: E402


def score_lists(candidates, targets, workers=1):
    lines = [c for spectrum in candidates for c in spectrum]
    return score(candidates, targets, canonicalize_all([*lines, *targets], workers))


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
