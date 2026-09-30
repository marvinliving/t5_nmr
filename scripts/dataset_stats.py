"""Token-length and tokenizer statistics for the NMR dataset (audit phase 0).

    python scripts/dataset_stats.py --model-name google/flan-t5-base

Needs only the tokenizer, no GPU. Reports, for each split:

- token-length percentiles of spectra (src) and SMILES (tgt);
- the share of SMILES longer than --target-max-length tokens, which training
  truncates and so can never predict exactly;
- SMILES that do not survive decode(encode(s)) == s, and the characters
  responsible. Such molecules can never score an exact match;
- padding overhead of random batches against length-grouped batches.

Results are printed and written as JSON to --output.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nmr_data import DEFAULT_PREFIX, SPLIT_FILES, read_lines  # noqa: E402


PERCENTILES = (50, 90, 99, 99.9, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", REPO_DIR / "alberts_2d")),
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(SPLIT_FILES), choices=list(SPLIT_FILES)
    )
    parser.add_argument("--target-max-length", type=int, default=640)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Use a random sample of this many rows per split; 0 uses all",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=REPO_DIR / "reports" / "dataset_stats.json"
    )
    return parser.parse_args()


def percentiles(lengths: np.ndarray) -> dict:
    return {f"p{p:g}": float(np.percentile(lengths, p)) for p in PERCENTILES}


def token_lengths(tokenizer, texts: list[str], batch: int = 10_000) -> np.ndarray:
    lengths = []
    for start in range(0, len(texts), batch):
        encoded = tokenizer(texts[start:start + batch], truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.array(lengths)


def round_trip(tokenizer, smiles: list[str], batch: int = 10_000) -> dict:
    """Which SMILES change when encoded and decoded, and which characters are lost."""
    failures = 0
    lost_characters = Counter()
    examples = []
    for start in range(0, len(smiles), batch):
        texts = smiles[start:start + batch]
        ids = tokenizer(texts, truncation=False)["input_ids"]
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        for original, result in zip(texts, decoded):
            if original == result:
                continue
            failures += 1
            if len(examples) < 20:
                examples.append({"smiles": original, "decoded": result})
            # Characters that appear less often after the round trip.
            before, after = Counter(original), Counter(result)
            for character, count in before.items():
                if after[character] < count:
                    lost_characters[character] += 1
    return {
        "failures": failures,
        "failure_rate": failures / len(smiles) if smiles else 0.0,
        "molecules_losing_character": dict(lost_characters.most_common()),
        "examples": examples,
    }


def padding_overhead(lengths: np.ndarray, batch_size: int, grouped: bool, seed: int) -> float:
    """Padded tokens / real tokens for one epoch of batches.

    grouped mimics the Trainer's LengthGroupedSampler: shuffle, cut into
    mega-batches of 50 batches, and sort each mega-batch by length.
    """
    order = np.random.default_rng(seed).permutation(len(lengths))
    if grouped:
        mega = 50 * batch_size
        order = np.concatenate(
            [
                chunk[np.argsort(-lengths[chunk], kind="stable")]
                for chunk in np.array_split(order, max(1, len(order) // mega))
            ]
        )
    padded = 0
    for start in range(0, len(order), batch_size):
        batch = lengths[order[start:start + batch_size]]
        padded += batch.max() * len(batch)
    return float(padded / lengths.sum() - 1)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, local_files_only=args.local_files_only
    )
    rng = random.Random(args.seed)
    report = {"model_name": args.model_name, "splits": {}}

    for split in args.splits:
        name = SPLIT_FILES[split]
        sources = read_lines(args.data_dir / f"src-{name}.txt")
        targets = read_lines(args.data_dir / f"tgt-{name}.txt")
        rows = list(range(len(sources)))
        if 0 < args.max_rows < len(rows):
            rows = sorted(rng.sample(rows, args.max_rows))
        sources = [args.prefix + sources[row] for row in rows]
        targets = [targets[row] for row in rows]
        print(f"===== {split}: {len(rows)} rows =====", flush=True)

        src_lengths = token_lengths(tokenizer, sources)
        tgt_lengths = token_lengths(tokenizer, targets)
        stats = {
            "rows": len(rows),
            "src_tokens": percentiles(src_lengths),
            "tgt_tokens": percentiles(tgt_lengths),
            "tgt_longer_than_target_max_length": float(
                (tgt_lengths > args.target_max_length).mean()
            ),
            "tgt_round_trip": round_trip(tokenizer, targets),
        }
        if split == "train":
            stats["padding_overhead"] = {
                f"batch_{size}": {
                    "random": padding_overhead(src_lengths, size, False, args.seed),
                    "length_grouped": padding_overhead(src_lengths, size, True, args.seed),
                }
                for size in args.batch_sizes
            }
        report["splits"][split] = stats
        print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
