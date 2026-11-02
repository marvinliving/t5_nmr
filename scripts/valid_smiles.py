"""Top-n exact match counting only the predictions that are valid SMILES.

    python scripts/valid_smiles.py outputs/<run>/prd-test.txt --data-dir /projects/b5an/alberts_2d

Reads a predictions file (N candidates per spectrum, best first, as written by
evaluate_exact_match.py and combine_results.py), the reference SMILES and the
molecular formula each input starts with, and checks every candidate with
RDKit. Reports:

- validity: the share of candidates RDKit can parse, overall and at each rank;
- formula: the share of candidates with the input's molecular formula;
- raw_exact: top-n exact match as the evaluation scores it, as a cross-check;
- valid_exact: top-n exact match after dropping invalid candidates, so the
  n-th valid candidate moves up to rank n;
- valid_canonical: as valid_exact, but comparing RDKit canonical SMILES, so the
  same molecule written differently counts, and repeated molecules are dropped;
- formula_canonical: as valid_canonical, but also dropping candidates whose
  molecular formula differs from the input's;
- formula_vote: the molecules of formula_canonical, ranked by how many
  candidates write them, ties kept in beam order.

An input without a formula keeps all its valid candidates in the formula
rankings.

If the predictions and the references write SMILES differently (as spaced
tokens or not), raw_exact and valid_exact remove the spaces from both before
comparing, as the evaluation does.

Accuracies are over all spectra, as in reports/evaluation_summary.tsv. Results
are printed and written as JSON next to the predictions. ./submit.sh valid
<run> runs this on a whole node, one worker per core.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nmr_data import (  # noqa: E402
    SPACINGS,
    SPLIT_FILES,
    read_lines,
    smiles_spacing,
    spacing_rule,
    without_spaces,
)

try:
    from rdkit import Chem, RDLogger
except ImportError:
    sys.exit("RDKit is not installed: pip install rdkit")

# RDKit logs every SMILES it cannot parse.
RDLogger.DisableLog("rdApp.*")

RANKINGS = ("raw_exact", "valid_exact", "valid_canonical", "formula_canonical", "formula_vote")

ELEMENT = re.compile(r"[A-Z][a-z]?")


class Molecule(NamedTuple):
    canonical: str
    formula: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("predictions", type=Path, help="prd-<split>.txt or a chunk's *_predictions.txt")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First reference the predictions belong to, for a chunk's predictions file",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=0,
        help="Candidates per spectrum; 0 infers it from the number of lines",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Processes for RDKit; 0 uses every CPU the job has, 1 runs without a pool",
    )
    parser.add_argument(
        "--smiles-spacing",
        choices=("auto", *SPACINGS),
        default="auto",
        help=(
            "How the model wrote the predictions; auto reads it from them. If it "
            "differs from the references, spaces are removed from both before "
            "comparing."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON results; defaults to <predictions stem>_valid.json next to them",
    )
    return parser.parse_args()


def parse(smiles: str) -> Optional[Chem.Mol]:
    """RDKit molecule, or None if RDKit cannot parse it.

    The datasets store SMILES as space-separated tokens ("C C ( = O ) O").
    RDKit reads everything after the first space as the molecule's name, so
    the tokens are joined first.
    """
    smiles = "".join(smiles.split())
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def canonical(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or None if RDKit cannot parse it."""
    mol = parse(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def hill_formula(counts: Mapping[str, int]) -> str:
    """Molecular formula in Hill order: C, H, then alphabetical; all alphabetical without C."""
    first = [element for element in ("C", "H") if "C" in counts and element in counts]
    order = first + sorted(element for element in counts if element not in first)
    return "".join(element + (str(counts[element]) if counts[element] > 1 else "") for element in order)


def input_formula(source: str) -> Optional[str]:
    """The molecular formula an input line starts with, or None if it has none.

    It is written as spaced tokens, each element followed by its count unless
    that is 1: "C 11 H 14 O S 3 HNMR 7.27 d ..." is C11H14OS3. It ends at the
    first token that is neither, such as HNMR or 1HNMR.
    """
    counts = Counter()
    tokens = source.split()
    position = 0
    while position < len(tokens) and ELEMENT.fullmatch(tokens[position]):
        element = tokens[position]
        position += 1
        count = 1
        if position < len(tokens) and tokens[position].isdigit():
            count = int(tokens[position])
            position += 1
        counts[element] += count
    return hill_formula(counts) if counts else None


def mol_formula(mol: Chem.Mol) -> str:
    """Molecular formula of an RDKit molecule, with its hydrogens and without its charge."""
    counts = Counter()
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
        counts["H"] += atom.GetTotalNumHs()
    return hill_formula(+counts)


def molecule(smiles: str) -> Optional[Molecule]:
    """Canonical SMILES and molecular formula, or None if RDKit cannot parse it."""
    mol = parse(smiles)
    return None if mol is None else Molecule(Chem.MolToSmiles(mol), mol_formula(mol))


def canonicalize_all(smiles: Iterable[str], workers: int = 1) -> Dict[str, Optional[Molecule]]:
    """Canonical form and formula of every distinct SMILES, spread over workers processes."""
    unique = sorted(set(smiles))
    if workers <= 1:
        return {s: molecule(s) for s in unique}
    chunksize = max(1, min(1000, len(unique) // (workers * 4)))
    with Pool(workers) as pool:
        return dict(zip(unique, pool.imap(molecule, unique, chunksize=chunksize)))


def group_candidates(lines: Sequence[str], targets: int, num_outputs: int = 0) -> List[List[str]]:
    """Split prediction lines into num_outputs candidates per spectrum."""
    if num_outputs == 0:
        if targets == 0 or len(lines) % targets:
            raise ValueError(
                f"{len(lines)} prediction lines is not a whole number of "
                f"candidates for {targets} spectra; pass --num-outputs"
            )
        num_outputs = len(lines) // targets
    if len(lines) != targets * num_outputs:
        raise ValueError(
            f"{len(lines)} prediction lines, expected {targets} spectra x "
            f"{num_outputs} candidates"
        )
    return [lines[i:i + num_outputs] for i in range(0, len(lines), num_outputs)]


def count_match(
    top_n_matches: List[int],
    candidates: Sequence[str],
    target: str,
    ignore_spaces: bool = False,
) -> None:
    """Add one spectrum to top_n_matches[n - 1]: target in the first n candidates.

    The rule of evaluate_exact_match.count_match, repeated here so this
    CPU-only script doesn't import torch.
    """
    if ignore_spaces:
        target = without_spaces(target)
        candidates = [without_spaces(candidate) for candidate in candidates]
    if target in candidates:
        for rank in range(list(candidates).index(target), len(top_n_matches)):
            top_n_matches[rank] += 1


def first_unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def by_votes(items: Iterable[str]) -> List[str]:
    """Distinct items, the most frequent first; ties keep the order they first appear in."""
    votes = Counter(items)
    return sorted(votes, key=lambda item: -votes[item])


def score(
    candidates_per_spectrum: Sequence[Sequence[str]],
    targets: Sequence[str],
    molecules: Dict[str, Optional[Molecule]],
    formulas: Sequence[Optional[str]],
    ignore_spaces: bool = False,
) -> Dict[str, object]:
    """Validity, formula and top-n exact match counts; molecules maps every SMILES.

    formulas holds each input's molecular formula, None where it has none.
    ignore_spaces applies to raw_exact and valid_exact; canonical SMILES
    have no spaces either way.
    """
    samples = len(targets)
    num_outputs = len(candidates_per_spectrum[0]) if samples else 0
    matches = {ranking: [0] * num_outputs for ranking in RANKINGS}
    valid_at_rank = [0] * num_outputs
    valid_candidates = 0
    any_valid = 0
    invalid_targets = 0
    right_formula_at_rank = [0] * num_outputs
    right_formula_candidates = 0
    any_right_formula = 0
    without_formula = 0
    wrong_formula_targets = 0

    for candidates, target, formula in zip(candidates_per_spectrum, targets, formulas):
        valid = [c for c in candidates if molecules[c] is not None]
        for rank, candidate in enumerate(candidates):
            if molecules[candidate] is not None:
                valid_at_rank[rank] += 1
                if molecules[candidate].formula == formula:
                    right_formula_at_rank[rank] += 1
        valid_candidates += len(valid)
        any_valid += bool(valid)
        if formula is None:
            without_formula += 1
            right_formula = valid
        else:
            right_formula = [c for c in valid if molecules[c].formula == formula]
            right_formula_candidates += len(right_formula)
            any_right_formula += bool(right_formula)

        count_match(matches["raw_exact"], list(candidates), target, ignore_spaces)
        count_match(matches["valid_exact"], valid, target, ignore_spaces)
        target_molecule = molecules[target]
        if target_molecule is None:
            invalid_targets += 1
            continue
        if formula is not None and target_molecule.formula != formula:
            wrong_formula_targets += 1
        count_match(
            matches["valid_canonical"],
            first_unique(molecules[c].canonical for c in valid),
            target_molecule.canonical,
        )
        count_match(
            matches["formula_canonical"],
            first_unique(molecules[c].canonical for c in right_formula),
            target_molecule.canonical,
        )
        count_match(
            matches["formula_vote"],
            by_votes(molecules[c].canonical for c in right_formula),
            target_molecule.canonical,
        )

    def share(count: int) -> float:
        return count / samples if samples else 0.0

    with_formula = samples - without_formula
    return {
        "samples": samples,
        "num_outputs": num_outputs,
        "validity": {
            "valid_candidates": valid_candidates / (samples * num_outputs) if samples else 0.0,
            "valid_at_rank": [share(count) for count in valid_at_rank],
            "top_1_valid": share(valid_at_rank[0]) if num_outputs else 0.0,
            "any_valid": share(any_valid),
            "mean_valid_per_spectrum": share(valid_candidates),
            "invalid_references": invalid_targets,
        },
        # Over the spectra whose input has a formula.
        "formula": {
            "inputs_without_formula": without_formula,
            "right_formula_candidates": (
                right_formula_candidates / (with_formula * num_outputs) if with_formula else 0.0
            ),
            "right_formula_at_rank": [
                count / with_formula if with_formula else 0.0 for count in right_formula_at_rank
            ],
            "top_1_right_formula": (
                right_formula_at_rank[0] / with_formula if with_formula and num_outputs else 0.0
            ),
            "any_right_formula": any_right_formula / with_formula if with_formula else 0.0,
            "references_wrong_formula": wrong_formula_targets,
        },
        "top_n_matches": matches,
        "top_n_exact_match": {
            ranking: {f"top_{rank}": share(count) for rank, count in enumerate(counts, start=1)}
            for ranking, counts in matches.items()
        },
    }


def print_report(result: Dict[str, object]) -> None:
    validity = result["validity"]
    print(f"Spectra: {result['samples']}, candidates each: {result['num_outputs']}")
    print(f"Valid candidates: {validity['valid_candidates']:.4f}")
    print(f"Top-1 valid: {validity['top_1_valid']:.4f}")
    print(f"At least one valid: {validity['any_valid']:.4f}")
    print(f"Valid per spectrum: {validity['mean_valid_per_spectrum']:.2f}")
    print(f"References RDKit cannot parse: {validity['invalid_references']}")
    formula = result["formula"]
    print(f"Inputs without a formula: {formula['inputs_without_formula']}")
    print(f"Right formula candidates: {formula['right_formula_candidates']:.4f}")
    print(f"Top-1 right formula: {formula['top_1_right_formula']:.4f}")
    print(f"At least one right formula: {formula['any_right_formula']:.4f}")
    print(f"References with another formula than the input: {formula['references_wrong_formula']}")
    print()
    print(
        f"{'rank':>4}  {'valid':>7}  {'formula':>7}  "
        + "  ".join(f"{name:>17}" for name in RANKINGS)
    )
    for rank in range(result["num_outputs"]):
        accuracies = [
            result["top_n_exact_match"][name][f"top_{rank + 1}"] for name in RANKINGS
        ]
        print(
            f"{rank + 1:>4}  {validity['valid_at_rank'][rank]:>7.4f}  "
            f"{formula['right_formula_at_rank'][rank]:>7.4f}  "
            + "  ".join(f"{accuracy:>17.4f}" for accuracy in accuracies)
        )


def main() -> None:
    args = parse_args()
    targets_path = args.data_dir / f"tgt-{SPLIT_FILES[args.split]}.txt"
    sources_path = args.data_dir / f"src-{SPLIT_FILES[args.split]}.txt"
    lines = args.predictions.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in lines]
    all_targets = read_lines(targets_path)

    if args.num_outputs:
        count = len(lines) // args.num_outputs
    else:
        count = len(all_targets) - args.start_index
    targets = all_targets[args.start_index:args.start_index + count]
    sources = read_lines(sources_path)[args.start_index:args.start_index + count]
    if len(sources) != len(targets):
        sys.exit(f"{sources_path} has {len(sources)} inputs for {len(targets)} references")
    formulas = [input_formula(source) for source in sources]
    candidates = group_candidates(lines, len(targets), args.num_outputs)
    if args.smiles_spacing == "auto":
        model_smiles, model_source = smiles_spacing(lines), "from the predictions"
    else:
        model_smiles, model_source = args.smiles_spacing, "set by SMILES_SPACING"
    reference_smiles = smiles_spacing(targets)
    ignore_spaces, spacing_line = spacing_rule(model_smiles, model_source, reference_smiles)

    # The CPUs Slurm gave the job; os.cpu_count() where affinity isn't available.
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = args.workers or available or 1
    print("Predictions:", args.predictions)
    print("References:", targets_path, f"from {args.start_index}")
    print("Inputs (for the formulas):", sources_path)
    print("Workers:", workers)
    print(spacing_line, flush=True)
    molecules = canonicalize_all([*lines, *targets], workers)

    result = score(candidates, targets, molecules, formulas, ignore_spaces)
    print_report(result)

    output = args.output or args.predictions.with_name(f"{args.predictions.stem}_valid.json")
    result = {
        "predictions_file": str(args.predictions),
        "references_file": str(targets_path),
        "inputs_file": str(sources_path),
        "start_index": args.start_index,
        "model_spacing": model_smiles,
        "reference_spacing": reference_smiles,
        "ignore_spaces": ignore_spaces,
        **result,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Results:", output)


if __name__ == "__main__":
    main()
