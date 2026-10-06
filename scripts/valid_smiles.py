"""Top-n exact match counting only the predictions that are valid SMILES.

    python scripts/valid_smiles.py outputs/<run>/prd-test.txt --data-dir /projects/b5an/alberts_2d

Reads a predictions file (N candidates per spectrum, best first, as written by
evaluate_exact_match.py and combine_results.py) and the reference SMILES, and
checks every candidate with RDKit. Reports:

- validity: the share of candidates RDKit can parse, overall and at each rank;
- raw_exact: top-n exact match as the evaluation scores it, as a cross-check;
- valid_exact: top-n exact match after dropping invalid candidates, so the
  n-th valid candidate moves up to rank n;
- valid_canonical: as valid_exact, but comparing RDKit canonical SMILES, so the
  same molecule written differently counts, and repeated molecules are dropped.

Accuracies are over all spectra, as in reports/evaluation_summary.tsv. Results
are printed and written as JSON next to the predictions. ./submit.sh valid
<run> runs this on a whole node, one worker per core.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nmr_data import SPLIT_FILES, read_lines  # noqa: E402

try:
    from rdkit import Chem, RDLogger
except ImportError:
    sys.exit("RDKit is not installed: pip install rdkit")

# RDKit logs every SMILES it cannot parse.
RDLogger.DisableLog("rdApp.*")

RANKINGS = ("raw_exact", "valid_exact", "valid_canonical")


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
        "--output",
        type=Path,
        default=None,
        help="JSON results; defaults to <predictions stem>_valid.json next to them",
    )
    return parser.parse_args()


def canonical(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES, or None if RDKit cannot parse it."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canonicalize_all(smiles: Iterable[str], workers: int = 1) -> Dict[str, Optional[str]]:
    """Canonical form of every distinct SMILES, spread over workers processes."""
    unique = sorted(set(smiles))
    if workers <= 1:
        return {s: canonical(s) for s in unique}
    chunksize = max(1, min(1000, len(unique) // (workers * 4)))
    with Pool(workers) as pool:
        return dict(zip(unique, pool.imap(canonical, unique, chunksize=chunksize)))


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


def count_match(top_n_matches: List[int], candidates: Sequence[str], target: str) -> None:
    """Add one spectrum to top_n_matches[n - 1]: target in the first n candidates.

    The rule of evaluate_exact_match.count_match, repeated here so this
    CPU-only script doesn't import torch.
    """
    if target in candidates:
        for rank in range(list(candidates).index(target), len(top_n_matches)):
            top_n_matches[rank] += 1


def first_unique(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def score(
    candidates_per_spectrum: Sequence[Sequence[str]],
    targets: Sequence[str],
    canonical_of: Dict[str, Optional[str]],
) -> Dict[str, object]:
    """Validity and top-n exact match counts; canonical_of maps every SMILES."""
    samples = len(targets)
    num_outputs = len(candidates_per_spectrum[0]) if samples else 0
    matches = {ranking: [0] * num_outputs for ranking in RANKINGS}
    valid_at_rank = [0] * num_outputs
    valid_candidates = 0
    any_valid = 0
    invalid_targets = 0

    for candidates, target in zip(candidates_per_spectrum, targets):
        valid = [c for c in candidates if canonical_of[c] is not None]
        for rank, candidate in enumerate(candidates):
            if canonical_of[candidate] is not None:
                valid_at_rank[rank] += 1
        valid_candidates += len(valid)
        any_valid += bool(valid)

        count_match(matches["raw_exact"], list(candidates), target)
        count_match(matches["valid_exact"], valid, target)
        target_canonical = canonical_of[target]
        if target_canonical is None:
            invalid_targets += 1
        else:
            count_match(
                matches["valid_canonical"],
                first_unique(canonical_of[c] for c in valid),
                target_canonical,
            )

    def share(count: int) -> float:
        return count / samples if samples else 0.0

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
    print()
    print(f"{'rank':>4}  {'valid':>7}  " + "  ".join(f"{name:>15}" for name in RANKINGS))
    for rank in range(result["num_outputs"]):
        accuracies = [
            result["top_n_exact_match"][name][f"top_{rank + 1}"] for name in RANKINGS
        ]
        print(
            f"{rank + 1:>4}  {validity['valid_at_rank'][rank]:>7.4f}  "
            + "  ".join(f"{accuracy:>15.4f}" for accuracy in accuracies)
        )


def main() -> None:
    args = parse_args()
    targets_path = args.data_dir / f"tgt-{SPLIT_FILES[args.split]}.txt"
    lines = args.predictions.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in lines]
    all_targets = read_lines(targets_path)

    if args.num_outputs:
        count = len(lines) // args.num_outputs
    else:
        count = len(all_targets) - args.start_index
    targets = all_targets[args.start_index:args.start_index + count]
    candidates = group_candidates(lines, len(targets), args.num_outputs)

    # The CPUs Slurm gave the job; os.cpu_count() where affinity isn't available.
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = args.workers or available or 1
    print("Predictions:", args.predictions)
    print("References:", targets_path, f"from {args.start_index}")
    print("Workers:", workers, flush=True)
    canonical_of = canonicalize_all([*lines, *targets], workers)

    result = score(candidates, targets, canonical_of)
    print_report(result)

    output = args.output or args.predictions.with_name(f"{args.predictions.stem}_valid.json")
    result = {
        "predictions_file": str(args.predictions),
        "references_file": str(targets_path),
        "start_index": args.start_index,
        **result,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Results:", output)


if __name__ == "__main__":
    main()
