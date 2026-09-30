"""Exact-match evaluation for an already trained FLAN-T5 model."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Tuple

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nmr_data import DEFAULT_PREFIX, build_generation_collate, load_split


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exact string match without training the model."
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help=(
            "Candidate SMILES to generate per spectrum, most likely first. "
            "1 uses greedy decoding; larger values use beam search and "
            "report top-1 to top-N exact match."
        ),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "alberts_2d",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First example to evaluate, using zero-based indexing.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Examples to evaluate; 0 means all examples after start-index.",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
        help=(
            "Where to write predictions, --num-outputs lines per spectrum. "
            "Defaults to the output file name with _predictions.txt."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help=(
            "Write predictions and partial counts every N batches. Spectra "
            "are sorted by length within each group of N batches."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from the spectra already in the predictions file "
            "instead of starting again."
        ),
    )
    parser.add_argument(
        "--no-sort-by-length",
        dest="sort_by_length",
        action="store_false",
        help=(
            "Generate in dataset order. Sorting by length cuts padding and "
            "is much faster; predictions are written in dataset order "
            "either way."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {args.model_path}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    if args.num_outputs <= 0:
        raise ValueError("--num-outputs must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")


def choose_range(
    total_count: int,
    start_index: int,
    sample_size: int,
) -> Tuple[int, int]:
    if total_count == 0:
        raise ValueError("The selected split is empty")
    if start_index >= total_count:
        raise ValueError(
            f"--start-index {start_index} is outside a split containing "
            f"{total_count} examples"
        )
    if sample_size == 0:
        end_index = total_count
    else:
        end_index = min(start_index + sample_size, total_count)
    return start_index, end_index


def default_output_file(
    model_path: Path,
    split: str,
    start_index: int,
    end_index: int,
    total_count: int,
) -> Path:
    output_dir = (
        model_path.parent if model_path.name == "final_model" else model_path
    )
    if start_index == 0 and end_index == total_count:
        filename = f"full_{split}_results.json"
    else:
        filename = f"{split}_{start_index}_{end_index}_results.json"
    return output_dir / filename


def default_predictions_file(output_file: Path) -> Path:
    stem = output_file.stem.removesuffix("_results")
    return output_file.with_name(f"{stem}_predictions.txt")


def top_n_exact_match(
    top_n_matches: Sequence[int],
    samples: int,
) -> Dict[str, float]:
    return {
        f"top_{rank}": count / samples
        for rank, count in enumerate(top_n_matches, start=1)
    }


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def count_match(
    top_n_matches: List[int],
    candidates: Sequence[str],
    target: str,
) -> None:
    """Add one spectrum to top_n_matches[n - 1]: target in the first n outputs."""
    if target in candidates:
        for rank in range(candidates.index(target), len(top_n_matches)):
            top_n_matches[rank] += 1


def run_evaluation(
    model,
    tokenizer,
    dataset: Dataset,
    *,
    device: torch.device,
    prefix: str = DEFAULT_PREFIX,
    batch_size: int = 16,
    num_outputs: int = 1,
    max_new_tokens: int = 640,
    sort_by_length: bool = True,
    window_batches: int = 100,
    predictions_handle: Optional[TextIO] = None,
    on_window: Optional[Callable[[int, List[int]], None]] = None,
) -> List[int]:
    """Generate num_outputs SMILES per spectrum and count top-n exact matches.

    dataset has "src" and "tgt" columns. It is processed in windows of
    window_batches batches. Within a window, spectra are sorted by length so
    each batch pads to similar lengths; predictions are written to
    predictions_handle in dataset order at the end of each window, and
    on_window(processed, top_n_matches) is called after each write.

    Returns top_n_matches, where top_n_matches[n - 1] counts spectra whose
    reference is among the first n outputs.
    """
    collate = build_generation_collate(tokenizer, prefix)
    top_n_matches = [0] * num_outputs
    window_size = batch_size * window_batches
    processed = 0

    model.eval()
    progress = tqdm(
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="Generating",
        unit="batch",
    )
    with torch.inference_mode():
        for window_start in range(0, len(dataset), window_size):
            window = dataset.select(
                range(window_start, min(window_start + window_size, len(dataset)))
            )
            order = list(range(len(window)))
            if sort_by_length:
                sources = list(window["src"])
                order.sort(key=lambda index: len(sources[index]), reverse=True)

            candidates_by_index: Dict[int, List[str]] = {}
            for batch_start in range(0, len(order), batch_size):
                indices = order[batch_start:batch_start + batch_size]
                encoded, _ = collate([window[index] for index in indices])
                encoded = {
                    name: tensor.to(device) for name, tensor in encoded.items()
                }
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_outputs,
                    num_return_sequences=num_outputs,
                )
                predictions = [
                    prediction.strip()
                    for prediction in tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )
                ]
                # Beam search returns each spectrum's outputs together, best first.
                for position, index in enumerate(indices):
                    candidates_by_index[index] = predictions[
                        position * num_outputs:(position + 1) * num_outputs
                    ]
                progress.update(1)

            targets = list(window["tgt"])
            for index in range(len(window)):
                candidates = candidates_by_index[index]
                count_match(top_n_matches, candidates, targets[index].strip())
                if predictions_handle is not None:
                    predictions_handle.writelines(
                        candidate + "\n" for candidate in candidates
                    )
            if predictions_handle is not None:
                predictions_handle.flush()
            processed += len(window)
            if on_window is not None:
                on_window(processed, top_n_matches)

    progress.close()
    return top_n_matches


def resume_predictions(
    predictions_file: Path,
    targets: Sequence[str],
    num_outputs: int,
) -> Tuple[int, List[int]]:
    """Score the spectra already in predictions_file and return (count, matches).

    A job killed while writing can leave a spectrum with only some of its
    lines, or half a line; the file is cut back to the last whole spectrum.
    """
    top_n_matches = [0] * num_outputs
    if not predictions_file.is_file():
        return 0, top_n_matches

    with predictions_file.open("r", encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    # The text after the last newline is an unfinished line, or "".
    complete_lines = lines[:-1]
    done = min(len(complete_lines) // num_outputs, len(targets))
    kept = complete_lines[:done * num_outputs]

    with predictions_file.open("w", encoding="utf-8") as handle:
        handle.writelines(line + "\n" for line in kept)

    for index in range(done):
        candidates = kept[index * num_outputs:(index + 1) * num_outputs]
        count_match(top_n_matches, candidates, targets[index].strip())
    return done, top_n_matches


def choose_dtype(device: torch.device) -> torch.dtype:
    # T5 overflows in fp16, so without bf16 support fall back to fp32.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    validate_args(args)

    full_dataset = load_split(args.data_dir, args.split)
    total_count = len(full_dataset)
    start_index, end_index = choose_range(
        total_count,
        args.start_index,
        args.sample_size,
    )
    selected_dataset = full_dataset.select(range(start_index, end_index))
    selected_count = end_index - start_index

    output_file = args.output_file or default_output_file(
        args.model_path,
        args.split,
        start_index,
        end_index,
        total_count,
    )
    partial_file = output_file.with_suffix(".partial.json")
    predictions_file = args.predictions_file or default_predictions_file(
        output_file
    )
    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    resumed = 0
    resumed_matches = [0] * args.num_outputs
    if args.resume:
        resumed, resumed_matches = resume_predictions(
            predictions_file, list(selected_dataset["tgt"]), args.num_outputs
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = choose_dtype(device)
    slurm_job_id = os.environ.get(
        "SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")
    )

    print("===== Exact-match evaluation =====", flush=True)
    print("Model:", args.model_path, flush=True)
    print("Split:", args.split, flush=True)
    print(f"Range: [{start_index}, {end_index})", flush=True)
    print(f"Samples: {selected_count} / {total_count}", flush=True)
    print("Already done (resume):", resumed, flush=True)
    print("Outputs per spectrum:", args.num_outputs, flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Sort by length:", args.sort_by_length, flush=True)
    print("Device:", device, flush=True)
    print("Dtype:", model_dtype, flush=True)
    print("Output:", output_file, flush=True)
    print("Predictions:", predictions_file, flush=True)

    started_at = time.monotonic()

    def top_n_totals(matches: Sequence[int]) -> List[int]:
        return [done + new for done, new in zip(resumed_matches, matches)]

    def save_partial(processed: int, matches: List[int]) -> None:
        samples = resumed + processed
        totals = top_n_totals(matches)
        write_json(
            partial_file,
            {
                "status": "in_progress",
                "split": args.split,
                "start_index": start_index,
                "end_index": start_index + samples,
                "requested_end_index": end_index,
                "samples": samples,
                "matches": totals[0],
                "exact_match": totals[0] / samples,
                "top_n_matches": totals,
                "top_n_exact_match": top_n_exact_match(totals, samples),
                "elapsed_seconds": time.monotonic() - started_at,
            },
        )

    new_matches = [0] * args.num_outputs
    if resumed < selected_count:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_path,
            dtype=model_dtype,
            local_files_only=args.local_files_only,
        )
        model.to(device)

        mode = "a" if resumed else "w"
        with predictions_file.open(mode, encoding="utf-8") as predictions_handle:
            new_matches = run_evaluation(
                model,
                tokenizer,
                selected_dataset.select(range(resumed, selected_count)),
                device=device,
                prefix=args.prefix,
                batch_size=args.batch_size,
                num_outputs=args.num_outputs,
                max_new_tokens=args.max_new_tokens,
                sort_by_length=args.sort_by_length,
                window_batches=args.save_every,
                predictions_handle=predictions_handle,
                on_window=save_partial,
            )

    top_n_matches = top_n_totals(new_matches)
    elapsed_seconds = time.monotonic() - started_at
    results: Dict[str, object] = {
        "status": "complete",
        "model_path": str(args.model_path.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "start_index": start_index,
        "end_index": end_index,
        "split_size": total_count,
        "samples": selected_count,
        "num_outputs": args.num_outputs,
        "matches": top_n_matches[0],
        "exact_match": top_n_matches[0] / selected_count,
        "top_n_matches": top_n_matches,
        "top_n_exact_match": top_n_exact_match(top_n_matches, selected_count),
        "predictions_file": str(predictions_file.resolve()),
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "sort_by_length": args.sort_by_length,
        "resumed_samples": resumed,
        "slurm_job_id": slurm_job_id,
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(output_file, results)
    write_json(partial_file, results)

    print("===== Results =====", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    print("Saved to:", output_file, flush=True)
    print("Predictions saved to:", predictions_file, flush=True)


if __name__ == "__main__":
    main()
