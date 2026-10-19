"""Score every saved checkpoint of a run, to see whether more training would help.

    python scripts/checkpoint_progress.py outputs/xl_4x1x4_10ep \
        --base-model google/flan-t5-xl --data-dir /projects/b5an/nmr_expt_data --split test

Generates SMILES for the first molecules of a split (1,000 by default, or all
of them) with each checkpoint-<step> in the output folder and with
final_model, and prints a table of exact match next to each one's epoch,
training loss and validation loss. If exact match is still rising from one
checkpoint to the next, the model is still learning; if it falls while the
training loss keeps falling, it is overtraining. The dataset can be the run's
own or another one; ./submit.sh progress (slurm/progress.sbatch) scores the
test split of the external dataset, nmr_expt_data, by default.

--num-outputs 1 is greedy top-1. A larger value uses beam search and scores
top-1 to top-N, for example --num-outputs 10. If the run was trained on
no-space SMILES and the dataset stores them as spaced tokens, or the reverse,
spaces are removed before comparing (--smiles-spacing).

FSDP checkpoints hold one shard per GPU; they are read straight into one
model, so each model needs one GPU whatever the run trained on. With --task i
--tasks n, a process scores only every n-th model still to do, so
progress.sbatch runs one per GPU, each on its own share; --table-only then
prints the table of all of them.

Results are kept in <output dir>/progress/<dataset>_<split>/ (with _top<N>
for N > 1) and reused, so running it again only scores new checkpoints, and a
checkpoint's score stays after training deletes the checkpoint itself. A
model whose scoring was interrupted is resumed from its predictions file.
Both losses are read again from trainer_state.json (see loss_curve.py) each
time the table is built.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from evaluate_exact_match import (  # noqa: E402
    choose_dtype,
    resume_predictions,
    run_evaluation,
    top_n_exact_match,
)
from loss_curve import eval_loss_at, mean_train_loss, read_history  # noqa: E402
from nmr_data import (  # noqa: E402
    DEFAULT_PREFIX,
    SPACINGS,
    load_split,
    model_spacing,
    spacing_rule,
    target_spacing,
)
from t5_train import final_model_is_complete, fix_t5_embeddings  # noqa: E402

# Where accelerate saves an FSDP SHARDED_STATE_DICT checkpoint's model.
FSDP_MODEL_DIR = "pytorch_model_fsdp_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path, help="The run's output folder")
    parser.add_argument(
        "--base-model",
        required=True,
        help="MODEL_NAME of the run; FSDP checkpoints take its config and tokenizer",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Molecules to score, the first ones of the split; 0 uses all",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help=(
            "Candidate SMILES per spectrum: 1 is greedy top-1; N > 1 is beam "
            "search, scored top-1 to top-N"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=100_000,
        help=(
            "Cap on batch rows x --num-outputs x longest input tokens, which "
            "sets the size of the cross-attention cache; long spectra get "
            "smaller batches than --batch-size. 0 means no cap."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--task",
        type=int,
        default=0,
        help=(
            "Score only every n-th model still to do, starting at the i-th "
            "(i = --task, n = --tasks); progress.sbatch runs one per GPU"
        ),
    )
    parser.add_argument("--tasks", type=int, default=1, help="See --task")
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Print and save the table of models already scored, without scoring",
    )
    parser.add_argument(
        "--smiles-spacing",
        choices=("auto", *SPACINGS),
        default="auto",
        help=(
            "How the run's model writes SMILES. auto reads it from the training "
            "targets named in its run_config.json. If it differs from the "
            "references, spaces are removed from both before comparing."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.task < args.tasks:
        parser.error(f"--task must be from 0 to --tasks - 1, got {args.task} of {args.tasks}")
    if args.num_outputs < 1:
        parser.error("--num-outputs must be 1 or more")
    return args


def find_models(output_dir: Path) -> list[Path]:
    """Complete checkpoints in step order, then final_model if it is complete.

    The Trainer writes trainer_state.json last, so a checkpoint without it is
    still being saved.
    """
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.name.split("-")[-1].isdigit() and (path / "trainer_state.json").is_file()
        ),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    final_model = output_dir / "final_model"
    if final_model_is_complete(final_model) and (output_dir / "trainer_state.json").is_file():
        # With per-epoch saving, the last checkpoint holds the same weights.
        final_step = training_progress(final_model, output_dir, {})["step"]
        if not checkpoints or training_progress(checkpoints[-1], output_dir, {})["step"] != final_step:
            checkpoints.append(final_model)
    return checkpoints


def eval_losses(output_dir: Path) -> dict[int, float]:
    """Validation loss by step, from every trainer_state.json of the run."""
    return {
        entry["step"]: entry["eval_loss"]
        for entry in read_history(output_dir)
        if "eval_loss" in entry
    }


def training_progress(model_dir: Path, output_dir: Path, losses: dict[int, float]) -> dict:
    """Epoch and step of model_dir, and the validation loss measured at that step.

    Validation loss is measured at the end of each epoch, so a checkpoint
    saved mid-epoch has none.
    """
    state_path = model_dir / "trainer_state.json"
    if model_dir.name == "final_model":
        # The final trainer_state.json is written to the output folder.
        state_path = output_dir / "trainer_state.json"
    state = json.loads(state_path.read_text())
    return {
        "epoch": round(state.get("epoch") or 0, 2),
        "step": state["global_step"],
        "eval_loss": losses.get(state["global_step"]),
    }


def same_settings(result: dict, settings: dict) -> bool:
    """result was scored with settings.

    Results from before --num-outputs are top-1, and those from before SMILES
    spacing was checked compared the SMILES as written.
    """
    result = {"num_outputs": 1, "ignore_spaces": False, **result}
    return all(result.get(key) == value for key, value in settings.items())


def load_sharded_weights(model: torch.nn.Module, shard_dir: Path) -> None:
    """Read an FSDP SHARDED_STATE_DICT checkpoint into model, on one process.

    accelerate saves {"model": state_dict}, so the stored names start with
    "model.". Each tensor is copied into the model's own, converting fp32 to
    the model's dtype.
    """
    import torch.distributed.checkpoint as dcp

    stored = dcp.FileSystemReader(str(shard_dir)).read_metadata().state_dict_metadata
    state_dict = model.state_dict()
    targets = {name: tensor for name, tensor in state_dict.items() if f"model.{name}" in stored}
    # A name missing from the checkpoint is fine only if it shares its
    # tensor with one that is loaded, as the tied input embeddings do.
    loaded = {tensor.data_ptr() for tensor in targets.values()}
    missing = [
        name
        for name, tensor in state_dict.items()
        if name not in targets and tensor.data_ptr() not in loaded
    ]
    if missing:
        raise ValueError(f"{shard_dir} has no weights for: {', '.join(missing)}")
    dcp.load({"model": targets}, checkpoint_id=str(shard_dir), no_dist=True)


def load_model(
    model_dir: Path,
    base_model: str,
    dtype: torch.dtype,
    device: torch.device,
    local_files_only: bool,
) -> torch.nn.Module:
    if (model_dir / "config.json").is_file():
        # final_model, or a checkpoint from a run without FSDP.
        return AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=dtype).to(device)
    shard_dir = model_dir / FSDP_MODEL_DIR
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"{model_dir} has neither config.json nor {FSDP_MODEL_DIR}")
    config = AutoConfig.from_pretrained(base_model, local_files_only=local_files_only)
    with torch.device(device):
        model = AutoModelForSeq2SeqLM.from_config(config, dtype=dtype)
    # The structure training had: lm_head separate from the input embeddings.
    fix_t5_embeddings(model)
    load_sharded_weights(model, shard_dir)
    return model


def score_models(args: argparse.Namespace, progress_dir: Path, dataset, settings: dict) -> None:
    """Score this task's share of the models not yet scored with settings.

    Every task lists the same models within its first seconds, before any
    scoring finishes, so no two tasks pick the same model. A model with a
    .partial.json is still to do, and is resumed by whichever task gets it.
    """
    todo = []
    for model_dir in find_models(args.output_dir):
        result_path = progress_dir / f"{model_dir.name}.json"
        if result_path.is_file() and same_settings(json.loads(result_path.read_text()), settings):
            print(f"{model_dir.name}: already scored")
        else:
            todo.append(model_dir)
    mine = todo[args.task::args.tasks]
    print(
        f"Task {args.task} of {args.tasks}: scoring "
        f"{', '.join(model_dir.name for model_dir in mine) or 'nothing'}"
    )
    if not mine:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)
    targets = list(dataset["tgt"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=args.local_files_only
    )
    losses = eval_losses(args.output_dir)
    for model_dir in mine:
        print(f"===== {model_dir.name} =====")
        start = time.time()
        try:
            progress = training_progress(model_dir, args.output_dir, losses)
            model = load_model(
                model_dir, args.base_model, dtype, device, args.local_files_only
            )
        except FileNotFoundError as error:
            # Training deletes old checkpoints, possibly while this runs.
            print(f"Skipped {model_dir.name}: {error}")
            continue
        # Predictions are written as they are made, so a job stopped at its
        # time limit carries on from them when submitted again.
        predictions_path = progress_dir / f"{model_dir.name}_predictions.txt"
        partial_path = progress_dir / f"{model_dir.name}.partial.json"
        done, matches = 0, [0] * args.num_outputs
        if partial_path.is_file() and same_settings(json.loads(partial_path.read_text()), settings):
            done, matches = resume_predictions(
                predictions_path, targets, args.num_outputs, settings["ignore_spaces"]
            )
            print(f"Resuming after {done} of {len(dataset)} molecules")
        else:
            partial_path.write_text(json.dumps(settings, indent=2) + "\n")
            predictions_path.write_text("")
        with predictions_path.open("a", encoding="utf-8") as predictions:
            if done < len(dataset):
                new = run_evaluation(
                    model,
                    tokenizer,
                    dataset.select(range(done, len(dataset))),
                    device=device,
                    prefix=args.prefix,
                    batch_size=args.batch_size,
                    max_batch_tokens=args.max_batch_tokens,
                    num_outputs=args.num_outputs,
                    max_new_tokens=args.max_new_tokens,
                    predictions_handle=predictions,
                    ignore_spaces=settings["ignore_spaces"],
                )
                matches = [old + added for old, added in zip(matches, new)]
        del model
        torch.cuda.empty_cache()

        result = {
            "model": model_dir.name,
            **progress,
            "exact_match": matches[0] / len(dataset),
            **top_n_exact_match(matches, len(dataset)),
            **settings,
            "seconds": round(time.time() - start, 1),
        }
        (progress_dir / f"{model_dir.name}.json").write_text(json.dumps(result, indent=2) + "\n")
        partial_path.unlink()
        print(f"{model_dir.name}: exact match {result['exact_match']:.4f}")


def print_table(rows: list[dict], columns: list[str]) -> None:
    print("\t".join(columns + ["change"]))
    previous = None
    for row in rows:
        change = "" if previous is None else f"{row['exact_match'] - previous:+.4f}"
        previous = row["exact_match"]
        values = [
            "" if row.get(column) is None
            else f"{row[column]:.4f}" if isinstance(row[column], float) and column != "epoch"
            else str(row[column])
            for column in columns
        ]
        print("\t".join(values + [change]))


def write_table(args: argparse.Namespace, progress_dir: Path, settings: dict) -> None:
    """Print and save the table of every model scored with settings.

    It includes checkpoints that training has since deleted. Both losses are
    read again from trainer_state.json, since they may have been logged after
    the model was scored.
    """
    history = read_history(args.output_dir)
    rows = []
    for path in progress_dir.glob("*.json"):
        if path.name.endswith(".partial.json"):
            continue
        row = json.loads(path.read_text())
        if same_settings(row, settings):
            row["eval_loss"] = eval_loss_at(history, row["step"]) or row["eval_loss"]
            row["train_loss"] = mean_train_loss(history, row["epoch"])
            rows.append(row)
    rows.sort(key=lambda row: row["step"])
    if not rows:
        print("No checkpoints scored yet in", args.output_dir)
        return

    num_outputs = settings["num_outputs"]
    columns = [
        "model", "epoch", "step", "train_loss", "eval_loss", "exact_match",
        *(f"top_{rank}" for rank in range(2, num_outputs + 1)),
        "samples",
    ]
    with (progress_dir / "progress.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if num_outputs == 1:
        scored = "Exact match"
    else:
        scored = f"Top-1 to top-{num_outputs} exact match (beam search)"
    spaces = ", spaces ignored" if settings["ignore_spaces"] else ""
    print(
        f"\n===== {scored} on {settings['samples']} {args.split} molecules "
        f"of {args.data_dir}{spaces} ====="
    )
    print_table(rows, columns)
    print("Saved to:", progress_dir / "progress.tsv")


def main() -> None:
    args = parse_args()
    folder = f"{args.data_dir.resolve().name}_{args.split}"
    if args.num_outputs > 1:
        # Kept apart, so a top-10 run never replaces the top-1 scores.
        folder += f"_top{args.num_outputs}"
    progress_dir = args.output_dir / "progress" / folder
    progress_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_split(args.data_dir, args.split)
    if 0 < args.samples < len(dataset):
        dataset = dataset.select(range(args.samples))
    # A model trained on no-space SMILES would match none of a spaced test
    # set's references, or the reverse.
    ignore_spaces, spacing_line = spacing_rule(
        *model_spacing(args.output_dir, args.smiles_spacing),
        target_spacing(args.data_dir, args.split),
    )
    print(spacing_line)
    settings = {
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "samples": len(dataset),
        "max_new_tokens": args.max_new_tokens,
        "prefix": args.prefix,
        "num_outputs": args.num_outputs,
        "ignore_spaces": ignore_spaces,
    }

    if not args.table_only:
        score_models(args, progress_dir, dataset, settings)
    write_table(args, progress_dir, settings)


if __name__ == "__main__":
    main()
