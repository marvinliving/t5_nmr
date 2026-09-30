"""Check that no reference SMILES is longer than the model can learn or generate.

    python scripts/check_target_lengths.py /projects/b5an/alberts_2d /projects/b5an/nmr_expt_data

Two limits cut SMILES short, both counted in tokens:

- training: labels are truncated to TARGET_MAX_LENGTH tokens, including the
  end-of-sequence token, so a longer SMILES is taught with its tail missing;
- generation: evaluation stops after GENERATION_MAX_NEW_TOKENS tokens, so a
  longer SMILES can never be predicted exactly.

Tokenizes every tgt-*.txt of each dataset as t5_train.py does, prints the
longest SMILES and every row over a limit, and exits with status 1 if any row
is over. All FLAN-T5 sizes share one tokenizer, so one check covers them all.
Needs only the tokenizer, no GPU.
"""

import argparse
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from nmr_data import SPLIT_FILES, read_lines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dirs", type=Path, nargs="+", help="Dataset folders")
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument(
        "--target-max-length",
        type=int,
        default=int(os.environ.get("TARGET_MAX_LENGTH", 640)),
        help="Training label limit, with end-of-sequence (default 640)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("GENERATION_MAX_NEW_TOKENS", 640)),
        help="Generation limit (default 640)",
    )
    parser.add_argument(
        "--show", type=int, default=20, help="Rows over a limit to print per split"
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def target_lengths(tokenizer, smiles: list[str], batch: int = 10_000) -> list[int]:
    """Label length of each SMILES in tokens, end-of-sequence included."""
    lengths = []
    for start in range(0, len(smiles), batch):
        encoded = tokenizer(text_target=smiles[start:start + batch], truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def rows_over_limits(
    lengths: list[int], target_max_length: int, max_new_tokens: int
) -> tuple[list[int], list[int]]:
    """Rows truncated in training, and rows generation can't finish.

    Training keeps target_max_length tokens including end-of-sequence.
    Generation only needs the SMILES tokens: a prediction cut off just before
    its end-of-sequence token still decodes to the whole SMILES.
    """
    truncated = [row for row, n in enumerate(lengths) if n > target_max_length]
    unfinishable = [row for row, n in enumerate(lengths) if n - 1 > max_new_tokens]
    return truncated, unfinishable


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, local_files_only=args.local_files_only
    )
    print(
        f"Limits: training labels {args.target_max_length} tokens with end-of-sequence, "
        f"generation {args.max_new_tokens} new tokens"
    )

    failed = False
    for data_dir in args.data_dirs:
        print(f"\n===== {data_dir} =====")
        for split, name in SPLIT_FILES.items():
            path = data_dir / f"tgt-{name}.txt"
            if not path.is_file():
                print(f"{split}: no {path.name}, skipped")
                continue
            smiles = read_lines(path)
            lengths = target_lengths(tokenizer, smiles)
            truncated, unfinishable = rows_over_limits(
                lengths, args.target_max_length, args.max_new_tokens
            )
            longest = max(range(len(lengths)), key=lengths.__getitem__)
            print(
                f"{split}: {len(smiles)} SMILES, longest {lengths[longest]} tokens "
                f"(line {longest + 1}); over training limit: {len(truncated)}; "
                f"over generation limit: {len(unfinishable)}"
            )
            for row in sorted(set(truncated) | set(unfinishable))[: args.show]:
                shown = smiles[row] if len(smiles[row]) <= 100 else smiles[row][:100] + "..."
                print(f"  line {row + 1}: {lengths[row]} tokens  {shown}")
            failed = failed or bool(truncated or unfinishable)

    if failed:
        print("\nFAIL: some SMILES are longer than a limit")
        sys.exit(1)
    print("\nOK: every SMILES fits both limits")


if __name__ == "__main__":
    main()
