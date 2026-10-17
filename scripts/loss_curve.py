"""Training and validation loss of a run, from its trainer_state.json files.

    python scripts/loss_curve.py outputs/xxl_4x1x4_10ep

The Trainer logs the training loss every LOGGING_STEPS updates (the mean
since the previous log) and the validation loss at the end of every epoch.
These are the dicts printed in logs/train_<run>_*.out, and each is also
kept, at full precision and with its step, in the trainer_state.json saved
with every checkpoint and at the end of training. A checkpoint's history
covers only the jobs that led to it, so losses from failed jobs never
appear. Losses logged since the newest checkpoint of a run still training
aren't saved yet.

Writes loss_curve.tsv in the output folder, one row per logged step, and
prints the mean training loss and the validation loss of each epoch.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

TSV_COLUMNS = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]


def read_history(output_dir: Path) -> list[dict]:
    """Every training-loss and validation-loss entry of the run, in step order.

    Each trainer_state.json holds the history up to its own step. A
    checkpoint saved in the same step as an evaluation can be written
    before it, so all of them are merged; they are read oldest first, so
    the newest wins if two disagree. A file still being written, or deleted
    by training meanwhile, is skipped.
    """
    checkpoints = sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*/trainer_state.json")
            if path.parent.name.split("-")[-1].isdigit()
        ),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    entries = {}
    for path in [*checkpoints, output_dir / "trainer_state.json"]:
        try:
            history = json.loads(path.read_text())["log_history"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue
        for entry in history:
            for kind in ("loss", "eval_loss"):
                if kind in entry:
                    entries[(entry["step"], kind)] = entry
    return [entries[key] for key in sorted(entries)]


def mean_train_loss(history: list[dict], epoch: float) -> float | None:
    """Mean logged training loss over the epoch ending at epoch."""
    losses = [
        entry["loss"]
        for entry in history
        if "loss" in entry and epoch - 1 < entry["epoch"] <= epoch + 1e-6
    ]
    return sum(losses) / len(losses) if losses else None


def eval_loss_at(history: list[dict], step: int) -> float | None:
    """Validation loss logged at step; None if there was no evaluation then."""
    for entry in history:
        if entry["step"] == step and "eval_loss" in entry:
            return entry["eval_loss"]
    return None


def curve_rows(history: list[dict]) -> list[dict]:
    """One row per logged step; a training and a validation entry at the same step share it."""
    rows = {}
    for entry in history:
        row = rows.setdefault(entry["step"], {})
        row.update({column: entry[column] for column in TSV_COLUMNS if column in entry})
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", type=Path, help="The run's output folder")
    args = parser.parse_args()

    history = read_history(args.output_dir)
    if not history:
        sys.exit(
            f"No logged losses in {args.output_dir}: trainer_state.json is saved "
            "with each checkpoint and at the end of training"
        )

    path = args.output_dir / "loss_curve.tsv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(curve_rows(history))

    print("epoch\tstep\ttrain_loss\teval_loss")
    for entry in history:
        if "eval_loss" in entry:
            train_loss = mean_train_loss(history, entry["epoch"])
            print(
                f"{round(entry['epoch'], 2)}\t{entry['step']}\t"
                f"{'' if train_loss is None else f'{train_loss:.4f}'}\t"
                f"{entry['eval_loss']:.4f}"
            )
    train_entries = [entry for entry in history if "loss" in entry]
    if train_entries:
        last = train_entries[-1]
        print(
            f"Last training loss: {last['loss']:.4f} at step {last['step']}, "
            f"epoch {round(last['epoch'], 2)}"
        )
    print("Saved to:", path)


if __name__ == "__main__":
    main()
