"""Combine the chunk results of an evaluation array job into one score.

    python scripts/combine_results.py xl_4x4_10ep

Finds <output dir>/test_<start>_<end>_results.json for the run, checks that
the chunks cover the whole split exactly once, prints top-1 to top-N exact
match weighted by chunk size, joins the chunks' predictions into
prd-test.txt, and records top-1 to top-N in reports/evaluation_summary.tsv.
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nmr_data import split_size  # noqa: E402


SUMMARY_COLUMNS = [
    "experiment",
    "chunks",
    "total_samples",
    "exact_match",
    # Top-n exact match for n = 2 to 10, blank for runs with fewer outputs.
    *(f"top_{n}" for n in range(2, 11)),
    "job_id",
    "date",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="Run name, as in configs/train/<run>.env")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder with the chunk results; defaults to the run's OUTPUT_DIR",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--experiment",
        help="Name of the row in the summary; defaults to the run name",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "Dataset folder, only needed for results from older evaluations, "
            "which don't record the split size. Defaults to the run's "
            "DATA_DIR, then the DATA_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--total",
        type=int,
        help="Examples in the split, instead of reading it from the data",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_DIR / "reports" / "evaluation_summary.tsv",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Print the result without recording it in the summary",
    )
    return parser.parse_args()


def config_value(run: str, name: str) -> str | None:
    """A variable set in configs/train/<run>.env, or None."""
    config = REPO_DIR / "configs" / "train" / f"{run}.env"
    if not config.is_file():
        raise FileNotFoundError(f"No config {config}")
    for line in config.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def run_output_dir(run: str) -> Path:
    """OUTPUT_DIR from the run's config, or outputs/<run>."""
    return REPO_DIR / (config_value(run, "OUTPUT_DIR") or f"outputs/{run}")


def split_total(args: argparse.Namespace, chunks: list[dict]) -> int:
    """Examples in the split: given, recorded in the results, or counted."""
    if args.total:
        return args.total
    recorded = {chunk["split_size"] for chunk in chunks if "split_size" in chunk}
    if len(recorded) == 1 and all("split_size" in chunk for chunk in chunks):
        return recorded.pop()
    data_dir = args.data_dir or config_value(args.run, "DATA_DIR") or os.environ.get("DATA_DIR")
    if not data_dir:
        raise ValueError(
            "These results don't record the split size: pass --total or --data-dir"
        )
    return split_size(REPO_DIR / data_dir, args.split)


def load_chunks(output_dir: Path, split: str) -> list[dict]:
    pattern = re.compile(rf"{split}_(\d+)_(\d+)_results\.json")
    chunks = []
    for path in output_dir.iterdir():
        if pattern.fullmatch(path.name) or path.name == f"full_{split}_results.json":
            result = json.loads(path.read_text())
            result["path"] = path
            chunks.append(result)
    if not chunks:
        raise FileNotFoundError(f"No {split} chunk results in {output_dir}")
    return sorted(chunks, key=lambda chunk: chunk["start_index"])


def check_tiling(chunks: list[dict], total: int) -> None:
    """The chunks must cover [0, total) exactly once."""
    expected_start = 0
    for chunk in chunks:
        # Results from older versions of the evaluator have no status.
        if chunk.get("status", "complete") != "complete":
            raise ValueError(f"{chunk['path'].name} is not complete")
        if chunk["start_index"] != expected_start:
            raise ValueError(
                f"Chunks do not tile [0, {total}): expected a chunk starting at "
                f"{expected_start}, found {chunk['path'].name}. Remove results "
                "from earlier evaluations with different chunk sizes."
            )
        expected_start = chunk["end_index"]
    if expected_start != total:
        raise ValueError(
            f"Chunks cover [0, {expected_start}) but the split has {total} examples"
        )


def chunk_predictions_file(chunk: dict) -> Path:
    """The chunk's predictions file, next to its results."""
    # The results hold an absolute path, which breaks if the folder moved.
    name = Path(chunk.get("predictions_file", "")).name or (
        chunk["path"].name.removesuffix("_results.json") + "_predictions.txt"
    )
    return chunk["path"].with_name(name)


def check_spacing_rule(chunks: list[dict]) -> bool:
    """Whether the chunks compared SMILES without spaces; all must agree.

    Results from before the rule was recorded compared them as written.
    """
    rules = {chunk.get("ignore_spaces", False) for chunk in chunks}
    if len(rules) > 1:
        raise ValueError(
            "Some chunks compared SMILES without spaces and some as written. "
            "Submit the evaluation again: it re-scores every chunk from its "
            "predictions file."
        )
    return rules.pop()


def combine_predictions(chunks: list[dict], destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as output:
        for chunk in chunks:
            predictions = chunk_predictions_file(chunk)
            lines = predictions.read_text(encoding="utf-8").splitlines(keepends=True)
            expected = chunk["samples"] * chunk.get("num_outputs", 1)
            if len(lines) != expected:
                raise ValueError(
                    f"{predictions.name} has {len(lines)} lines, expected {expected}"
                )
            output.writelines(lines)


def record_summary(summary: Path, row: dict) -> None:
    rows = []
    if summary.is_file():
        with summary.open(newline="") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
    # Evaluating a run again replaces its row.
    rows = [existing for existing in rows if existing["experiment"] != row["experiment"]]
    rows.append(row)
    # Runs with more than 10 outputs add top_11 and beyond at the end.
    columns = list(SUMMARY_COLUMNS)
    for existing in rows:
        columns += [name for name in existing if name not in columns]
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=columns, delimiter="\t", restval="",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or run_output_dir(args.run)
    chunks = load_chunks(output_dir, args.split)
    total = split_total(args, chunks)
    check_tiling(chunks, total)
    if check_spacing_rule(chunks):
        print(
            "SMILES compared without spaces: the model and the references "
            "write them differently"
        )

    num_outputs = min(len(chunk.get("top_n_matches", [0])) for chunk in chunks)
    top_n = [
        sum(chunk.get("top_n_matches", [chunk["matches"]])[rank] for chunk in chunks)
        for rank in range(num_outputs)
    ]
    for rank, matches in enumerate(top_n, start=1):
        print(f"top-{rank}: {matches / total:.4f}")

    file_split = "val" if args.split == "validation" else "test"
    predictions = output_dir / f"prd-{file_split}.txt"
    if all(chunk_predictions_file(chunk).is_file() for chunk in chunks):
        combine_predictions(chunks, predictions)
        print("Predictions:", predictions)
    else:
        # Evaluations made before predictions were saved.
        print("Some chunks have no predictions file; prd file not written")

    if not args.no_summary:
        job_ids = sorted({str(chunk.get("slurm_job_id", "")) for chunk in chunks} - {""})
        record_summary(
            args.summary,
            {
                "experiment": args.experiment or args.run,
                "chunks": len(chunks),
                "total_samples": total,
                "exact_match": f"{top_n[0] / total:.8f}",
                **{
                    f"top_{rank}": f"{matches / total:.8f}"
                    for rank, matches in enumerate(top_n[1:], start=2)
                },
                "job_id": ",".join(job_ids),
                "date": datetime.date.today().isoformat(),
            },
        )
        print("Recorded in:", args.summary)


if __name__ == "__main__":
    main()
