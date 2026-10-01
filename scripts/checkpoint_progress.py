"""Score every saved checkpoint of a run, to see whether more training would help.

    python scripts/checkpoint_progress.py outputs/xl_4x1x4_10ep \
        --base-model google/flan-t5-xl --data-dir /projects/b5an/nmr_expt_data --split test

Generates SMILES for the first molecules of a split (or all of them) with
each checkpoint-<step> in the output folder and with final_model, and prints
a table of top-1 exact match next to each one's epoch and validation loss. If
exact match is still rising from one checkpoint to the next, the model is
still learning. The dataset can be the run's own or another one, such as an
external test set.

FSDP checkpoints hold one shard per GPU; they are read straight into one
model, so this needs one GPU whatever the run trained on. Results are kept in
<output dir>/progress/<dataset>_<split>/ and reused, so running it again only
scores new checkpoints, and a checkpoint's score stays after training
deletes the checkpoint itself. A model whose scoring was interrupted is
resumed from its predictions file.
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

from evaluate_exact_match import choose_dtype, resume_predictions, run_evaluation  # noqa: E402
from nmr_data import DEFAULT_PREFIX, load_split  # noqa: E402
from t5_train import final_model_is_complete, fix_t5_embeddings  # noqa: E402

# Where accelerate saves an FSDP SHARDED_STATE_DICT checkpoint's model.
FSDP_MODEL_DIR = "pytorch_model_fsdp_0"
TABLE_COLUMNS = ["model", "epoch", "step", "eval_loss", "exact_match", "samples"]


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
        default=2000,
        help="Molecules to score, the first ones of the split; 0 uses all",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


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
    """Validation loss by step, from every trainer_state.json of the run.

    A checkpoint saved in the same step as an evaluation can be written
    before it (at the last step of a MAX_STEPS run), so its own history may
    lack the loss that a later one records.
    """
    losses = {}
    for path in [*output_dir.glob("checkpoint-*/trainer_state.json"), output_dir / "trainer_state.json"]:
        if path.is_file():
            for entry in json.loads(path.read_text())["log_history"]:
                if "eval_loss" in entry:
                    losses[entry["step"]] = entry["eval_loss"]
    return losses


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


def print_table(rows: list[dict]) -> None:
    print("\t".join(TABLE_COLUMNS + ["change"]))
    previous = None
    for row in rows:
        change = "" if previous is None else f"{row['exact_match'] - previous:+.4f}"
        previous = row["exact_match"]
        values = [
            "" if row[column] is None
            else f"{row[column]:.4f}" if isinstance(row[column], float) and column != "epoch"
            else str(row[column])
            for column in TABLE_COLUMNS
        ]
        print("\t".join(values + [change]))


def main() -> None:
    args = parse_args()
    progress_dir = args.output_dir / "progress" / f"{args.data_dir.resolve().name}_{args.split}"
    progress_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype(device)

    dataset = load_split(args.data_dir, args.split)
    if 0 < args.samples < len(dataset):
        dataset = dataset.select(range(args.samples))
    targets = list(dataset["tgt"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=args.local_files_only
    )
    settings = {
        "data_dir": str(args.data_dir.resolve()),
        "split": args.split,
        "samples": len(dataset),
        "max_new_tokens": args.max_new_tokens,
        "prefix": args.prefix,
    }

    losses = eval_losses(args.output_dir)
    for model_dir in find_models(args.output_dir):
        result_path = progress_dir / f"{model_dir.name}.json"
        if result_path.is_file():
            previous = json.loads(result_path.read_text())
            if all(previous.get(key) == value for key, value in settings.items()):
                print(f"{model_dir.name}: already scored")
                continue

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
        done, matches = 0, [0]
        if partial_path.is_file() and json.loads(partial_path.read_text()) == settings:
            done, matches = resume_predictions(predictions_path, targets, 1)
            print(f"Resuming after {done} of {len(dataset)} molecules")
        else:
            partial_path.write_text(json.dumps(settings, indent=2) + "\n")
            predictions_path.write_text("")
        with predictions_path.open("a", encoding="utf-8") as predictions:
            if done < len(dataset):
                matches[0] += run_evaluation(
                    model,
                    tokenizer,
                    dataset.select(range(done, len(dataset))),
                    device=device,
                    prefix=args.prefix,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    predictions_handle=predictions,
                )[0]
        del model
        torch.cuda.empty_cache()

        result = {
            "model": model_dir.name,
            **progress,
            "exact_match": matches[0] / len(dataset),
            **settings,
            "seconds": round(time.time() - start, 1),
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        partial_path.unlink()
        print(f"{model_dir.name}: exact match {result['exact_match']:.4f}")

    # Every model scored with these settings, including checkpoints that
    # training has since deleted.
    losses = eval_losses(args.output_dir)
    rows = []
    for path in progress_dir.glob("*.json"):
        if path.name.endswith(".partial.json"):
            continue
        result = json.loads(path.read_text())
        if all(result.get(key) == value for key, value in settings.items()):
            # The loss may have been logged after the model was scored.
            result["eval_loss"] = losses.get(result["step"], result["eval_loss"])
            rows.append(result)
    rows.sort(key=lambda row: row["step"])
    if not rows:
        print("No checkpoints to score in", args.output_dir)
        return

    with (progress_dir / "progress.tsv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=TABLE_COLUMNS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"\n===== Exact match on {len(dataset)} {args.split} molecules "
        f"of {args.data_dir} ====="
    )
    print_table(rows)
    print("Saved to:", progress_dir / "progress.tsv")


if __name__ == "__main__":
    main()
