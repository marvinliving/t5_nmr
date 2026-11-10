"""Record training runs in reports/training_summary.tsv and their jobs in reports/training_jobs.tsv.

    python scripts/train_summary.py xl_4x1x4_10ep
    python scripts/train_summary.py --all

slurm/train.sbatch runs the first form at the end of every training job, so
a run's row follows it while it trains. --all records every run that has an
output folder or a training log in logs/, which fills in the runs trained
before this script existed. Rows of runs whose files are gone are kept.

A run's row has its model and parameter count, its dataset and the size of
each split, its batch and learning-rate settings, how far it got (epochs,
steps, losses), the quick test score, and what its jobs took (wall and GPU
hours). The settings come from run_config.json, or for runs from before it
existed, from the configuration their logs print; config_source says which.
Each job's row comes from its Slurm log: when it ran, on how many GPUs,
which steps it trained, and how it ended. Rows are named like those of
reports/evaluation_summary.tsv (combine_results.py), so the two join on
experiment.
"""

import argparse
import dataclasses
import datetime
import functools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from dateutil import parser as date_parser
from transformers import AutoConfig, AutoModelForSeq2SeqLM

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from combine_results import record_rows, run_output_dir  # noqa: E402
from config import TrainConfig  # noqa: E402
from loss_curve import mean_train_loss, read_history  # noqa: E402
from nmr_data import split_size  # noqa: E402
from t5_train import final_model_is_complete, fix_t5_embeddings  # noqa: E402


SUMMARY_COLUMNS = [
    "experiment",
    # complete (final_model saved), incomplete, or missing (no output folder).
    "status",
    "last_outcome",
    "model_name",
    "parameters",
    "model_gb",
    "finetune_from",
    "dataset",
    "train_rows",
    "validation_rows",
    "test_rows",
    # Only runs from before inputs were left untruncated have one.
    "input_max_length",
    "target_max_length",
    "epochs",
    "epochs_done",
    "steps_done",
    "total_steps",
    "per_gpu_batch",
    "grad_accumulation",
    "nodes",
    "gpus",
    "global_batch",
    "parallel_mode",
    "gradient_checkpointing",
    # The rate trained with, after LR_SCALING.
    "learning_rate",
    "lr_scaling",
    "warmup_ratio",
    "weight_decay",
    "seed",
    # Mean over the last epoch reached, and the last and lowest validation loss.
    "train_loss",
    "eval_loss",
    "best_eval_loss",
    "best_eval_epoch",
    "quick_test_exact_match",
    "quick_test_samples",
    "jobs",
    "failed_jobs",
    "wall_hours",
    "gpu_hours",
    "peak_gpu_mem_gb",
    "gpu_name",
    "started",
    "ended",
    "job_ids",
    # run_config.json, log, trainer_state, or blank if nothing records them.
    "config_source",
    "output_dir",
    "date",
]

JOB_COLUMNS = [
    "experiment",
    "job_id",
    "start",
    "end",
    "hours",
    "nodes",
    "gpus",
    "gpu_hours",
    "resubmission",
    "resumed_from_step",
    "stopped_at_step",
    "outcome",
    "peak_gpu_mem_gb",
    "gpu_name",
    "log_file",
]

# Printed by every training script so far, the current slurm/train.sbatch
# and the per-run scripts before it; evaluation and other jobs never print it.
TRAINING_MARKER = "===== Start Training ====="
# The marker and the output folder are in a log's header, before any
# progress bars.
HEAD_BYTES = 256 * 1024

# t5_train.py prints every TrainConfig setting as "key: value".
CONFIG_KEYS = {
    *(field.name for field in dataclasses.fields(TrainConfig)),
    "global_batch_size",
    "effective_learning_rate",
}
# What the training scripts from before configs/ printed instead.
OLD_LABELS = {
    "Model": "model_name",
    "Data directory": "data_dir",
    "Input max length": "input_max_length",
    "Target max length": "target_max_length",
    "Learning rate": "learning_rate",
    "Weight decay": "weight_decay",
    "Epochs": "num_epochs",
    "Train batch size per device": "train_batch_size",
    "Gradient accumulation steps": "grad_accumulation_steps",
    "Gradient checkpointing": "gradient_checkpointing",
    "Parallel mode": "parallel_mode",
    "World size (GPUs)": "world_size",
    "Max steps": "max_steps",
    "Random seed": "seed",
}
SPLIT_LABELS = {"Train rows": "train", "Validation rows": "validation", "Test rows": "test"}

# How a job ended, from text in its log; the first outcome found wins.
# finished and max_resubmits come from train.sbatch after training exits
# cleanly, so a traceback printed along the way doesn't count.
OUTCOMES = [
    ("already_finished", ("Training already finished",)),
    ("finished", ("===== Training finished =====", "===== Job Finished Successfully =====")),
    ("max_resubmits", ("MAX_RESUBMITS=",)),
    ("resubmitted", ("Stopped before the job time limit", "Resubmitting to continue")),
    ("timeout", ("DUE TO TIME LIMIT",)),
    ("out_of_memory", ("OutOfMemoryError", "CUDA out of memory", "oom-kill", "oom_kill")),
    ("cancelled", ("CANCELLED AT",)),
    ("failed", ("Traceback (most recent call last)", "srun: error:")),
]
FAILED_OUTCOMES = {"max_resubmits", "timeout", "out_of_memory", "cancelled", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", nargs="?", help="Run name, as in configs/train/<run>.env")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Record every run with an output folder or a training log",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="The run's output folder; defaults to the run's OUTPUT_DIR",
    )
    parser.add_argument(
        "--experiment",
        help=(
            "Name of the run's rows; defaults to the run name, or the folder "
            "name for a folder of its own, outputs/<run>_gb<N> or _max<N>steps"
        ),
    )
    parser.add_argument("--logs-dir", type=Path, default=REPO_DIR / "logs")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=REPO_DIR / "outputs",
        help="Where output folders named outputs/<name> in logs and configs are",
    )
    parser.add_argument(
        "--summary", type=Path, default=REPO_DIR / "reports" / "training_summary.tsv"
    )
    parser.add_argument(
        "--jobs", type=Path, default=REPO_DIR / "reports" / "training_jobs.tsv"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Print the runs without recording them",
    )
    args = parser.parse_args()
    if args.all == bool(args.run):
        parser.error("give a run or --all")
    if args.all and (args.output_dir or args.experiment):
        parser.error("--output-dir and --experiment are for one run")
    return args


def folder_key(path: str | Path) -> str:
    """An output folder as outputs/<name>, however a log or config names it.

    Old logs print the folder as an absolute path, newer ones relative to
    the repository. A folder outside any outputs/ is kept as an absolute path.
    """
    path = Path(path)
    if "outputs" in path.parts:
        start = len(path.parts) - 1 - path.parts[::-1].index("outputs")
        return Path(*path.parts[start:]).as_posix()
    return str(path if path.is_absolute() else REPO_DIR / path)


def key_folder(key: str, outputs_dir: Path) -> Path:
    """The folder a folder_key names, with outputs/ at outputs_dir."""
    if key.startswith("outputs/"):
        return outputs_dir / key.removeprefix("outputs/")
    return Path(key)


def experiment_name(run: str, output_dir: Path) -> str:
    """As slurm/combine.sbatch names it: the run, or the folder of a run with its own."""
    name = Path(output_dir).name
    return name if name.startswith(f"{run}_") else run


def configured_runs() -> dict[str, str]:
    """folder_key of every configs/train run's output folder -> run name."""
    return {
        folder_key(run_output_dir(config.stem)): config.stem
        for config in sorted((REPO_DIR / "configs" / "train").glob("*.env"))
    }


def parse_value(text: str):
    """A printed setting as the int, float, bool, None or string it was."""
    text = text.strip()
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(text, text)


def parse_time(text: str) -> datetime.datetime | None:
    """A time printed by date, in whichever locale the node used; None if unreadable."""
    try:
        return date_parser.parse(text, fuzzy=True, ignoretz=True)
    except (ValueError, OverflowError):
        return None


def log_output_key(path: Path) -> str | None:
    """The folder_key of a training log's output folder; None for other logs."""
    with path.open("rb") as file:
        head = file.read(HEAD_BYTES).decode("utf-8", errors="replace")
    if TRAINING_MARKER not in head:
        return None
    match = re.search(r"^Output directory: (.+)$", head, re.MULTILINE)
    return folder_key(match.group(1).strip()) if match else None


def parse_log(path: Path) -> dict:
    """What a training job's log says about the job.

    Read line by line: progress bars make logs of long jobs large. A bar
    redraws itself after a carriage return, so each line is split there too.
    """
    job = {
        "path": path,
        "job_id": None,
        "start": None,
        "end": None,
        "nodes": None,
        "gpus_per_node": None,
        "resubmission": None,
        "gpu_name": None,
        "resumed_from_step": None,
        "stopped_at_step": None,
        "peak_gpu_mem_gb": None,
        "settings": {},
        "split_rows": {},
    }
    found = set()
    patterns = [marker for _, markers in OUTCOMES for marker in markers]
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            for text in line.split("\r"):
                text = text.strip()
                # Progress bars, most of a long job's log, say nothing here.
                if not text or "%|" in text:
                    continue
                found.update(marker for marker in patterns if marker in text)
                read_line(job, text)
    for outcome, markers in OUTCOMES:
        if found.intersection(markers):
            job["outcome"] = outcome
            break
    else:
        job["outcome"] = None
    if job["end"] is None:
        # A job that didn't end cleanly stopped writing its log when it ended.
        job["end"] = datetime.datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0)
    return job


def read_line(job: dict, text: str) -> None:
    """Record what one line of a training log says in job."""
    if match := re.fullmatch(r"Job ID: (\d+)", text):
        job["job_id"] = job["job_id"] or match.group(1)
    elif text.startswith("Start time:"):
        job["start"] = job["start"] or parse_time(text.removeprefix("Start time:"))
    elif text.startswith("End time:"):
        job["end"] = parse_time(text.removeprefix("End time:")) or job["end"]
    elif match := re.fullmatch(r"Nodes: (?:.*\((\d+)\)|(\d+))", text):
        job["nodes"] = job["nodes"] or int(match.group(1) or match.group(2))
    elif match := re.fullmatch(r"GPUs per node: (\d+)", text):
        job["gpus_per_node"] = job["gpus_per_node"] or int(match.group(1))
    elif match := re.fullmatch(r"Resubmission: (\d+) of \d+", text):
        job["resubmission"] = int(match.group(1))
    elif match := re.fullmatch(r"GPU \d+: (.+)", text):
        job["gpu_name"] = job["gpu_name"] or match.group(1)
    elif match := re.search(r"Resuming from checkpoint: .*checkpoint-(\d+)", text):
        job["resumed_from_step"] = int(match.group(1))
    elif text.startswith(("Starting training from the pretrained model", "Fine-tuning ")):
        job["resumed_from_step"] = 0
    elif match := re.search(r"Stopped before the job time limit at step (\d+)", text):
        job["stopped_at_step"] = int(match.group(1))
    elif match := re.search(r"Peak GPU memory allocated \(GB\): ([\d.]+)", text):
        job["peak_gpu_mem_gb"] = float(match.group(1))
    elif match := re.fullmatch(r"([A-Za-z][\w ()/-]*?): (.*)", text):
        label, value = match.groups()
        if label in SPLIT_LABELS:
            job["split_rows"].setdefault(SPLIT_LABELS[label], parse_value(value))
        elif label in CONFIG_KEYS or label in OLD_LABELS:
            job["settings"].setdefault(OLD_LABELS.get(label, label), parse_value(value))


def still_queued(job_id: str) -> str | None:
    """The job's Slurm state, lower case, if squeue still lists it."""
    if not shutil.which("squeue"):
        return None
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().lower() or None


def job_outcome(job: dict) -> str:
    """How the job ended, or running if it hasn't."""
    if job["outcome"]:
        return job["outcome"]
    if job["job_id"] and job["job_id"] == os.environ.get("SLURM_JOB_ID"):
        # This job, recording itself on its way out without having printed
        # that it finished or will be resubmitted: train.sbatch failed.
        return "failed"
    if job["job_id"] and (state := still_queued(job["job_id"])):
        return state
    return "unknown"


def job_gpus(job: dict, run_world_size) -> int:
    """GPUs the job trained on.

    Logs from before multi-GPU training print no GPU count; those jobs had one.
    """
    world_size = job["settings"].get("world_size")
    if isinstance(world_size, int):
        return world_size
    if job["nodes"] and job["gpus_per_node"]:
        return job["nodes"] * job["gpus_per_node"]
    return run_world_size if isinstance(run_world_size, int) else 1


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest_trainer_state(output_dir: Path) -> dict | None:
    """The trainer_state.json that got furthest: the run's own or a checkpoint's."""
    states = [
        read_json(path)
        for path in [
            output_dir / "trainer_state.json",
            *output_dir.glob("checkpoint-*/trainer_state.json"),
        ]
    ]
    states = [state for state in states if state]
    return max(states, key=lambda state: state.get("global_step", 0), default=None)


@functools.lru_cache(maxsize=None)
def parameter_count(model: str) -> int | None:
    """Parameters of a model folder or Hugging Face model.

    The model is built on the meta device from its config alone, so no
    weights are read and even XXL takes a moment. Built from a config,
    transformers ties lm_head to the input embeddings; training unties it
    (fix_t5_embeddings), so it is untied here too, or it wouldn't count.
    """
    try:
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        with torch.device("meta"):
            built = AutoModelForSeq2SeqLM.from_config(config)
            fix_t5_embeddings(built)
        return sum(parameter.numel() for parameter in built.parameters())
    except Exception as error:  # Any failure only leaves the column blank.
        print(f"Parameters of {model} unknown: {error}", file=sys.stderr)
        return None


@functools.lru_cache(maxsize=None)
def dataset_rows(data_dir: str) -> dict:
    """Examples in each split of a dataset folder; splits it lacks are left out."""
    rows = {}
    for split in ("train", "validation", "test"):
        try:
            rows[split] = split_size(REPO_DIR / data_dir, split)
        except OSError:
            pass
    return rows


def cell(value, spec: str = "") -> str:
    """A TSV cell: blank for None, 1 or 0 for a flag, numbers formatted with spec."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if spec and isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def run_settings(output_dir: Path, jobs: list[dict], state: dict | None) -> tuple[dict, str]:
    """The run's settings, and what recorded them.

    run_config.json wins, then the logs, a newer one over an older one, then
    trainer_state.json. A run resumed after configs/ has both, and keeps the
    settings only its old logs print, such as the input length.
    """
    settings, source = {}, ""
    if state:
        settings.update(
            train_batch_size=state.get("train_batch_size"),
            num_epochs=state.get("num_train_epochs"),
        )
        source = "trainer_state"
    for job in jobs:
        if job["settings"]:
            settings.update(job["settings"])
            source = "log"
    run_config = read_json(output_dir / "run_config.json")
    if run_config:
        settings.update(run_config)
        source = "run_config.json"
    return settings, source


def summarise_run(experiment: str, output_dir: Path, jobs: list[dict]) -> tuple[dict, list[dict]]:
    """The run's summary row and one row per job."""
    jobs = sorted(jobs, key=lambda job: (job["start"] or datetime.datetime.min, job["job_id"] or ""))
    state = latest_trainer_state(output_dir)
    settings, source = run_settings(output_dir, jobs, state)

    world_size = settings.get("world_size")
    for job in jobs:
        job["gpus"] = job_gpus(job, world_size)
    if not isinstance(world_size, int) and jobs:
        world_size = jobs[-1]["gpus"]
    gpus_per_node = settings.get("gpus_per_node")
    nodes = next((job["nodes"] for job in reversed(jobs) if job["nodes"]), None)
    if nodes is None and isinstance(world_size, int) and gpus_per_node:
        nodes = world_size // gpus_per_node
    per_gpu = settings.get("train_batch_size")
    accumulation = settings.get("grad_accumulation_steps")
    global_batch = settings.get("global_batch_size")
    if global_batch is None and all(isinstance(value, int) for value in (per_gpu, accumulation, world_size)):
        global_batch = per_gpu * accumulation * world_size

    data_dir = settings.get("data_dir")
    split_rows = dataset_rows(str(data_dir)) if data_dir else {}
    for job in reversed(jobs):
        # From the log when the dataset can't be read here.
        split_rows = {**job["split_rows"], **split_rows}

    final_model = output_dir / "final_model"
    model_name = settings.get("model_name")
    parameters = None
    if (final_model / "config.json").is_file():
        parameters = parameter_count(str(final_model))
    elif model_name:
        parameters = parameter_count(model_name)
    weights = list(final_model.glob("*.safetensors"))

    history = read_history(output_dir) if output_dir.is_dir() else []
    train_entries = [entry for entry in history if "loss" in entry]
    eval_entries = [entry for entry in history if "eval_loss" in entry]
    train_loss = mean_train_loss(history, train_entries[-1]["epoch"]) if train_entries else None
    best = min(eval_entries, key=lambda entry: entry["eval_loss"], default=None)

    quick_test = read_json(output_dir / "test_results.json") or {}

    job_rows = []
    for job in jobs:
        gpus = job["gpus"]
        hours = None
        if job["start"] and job["end"]:
            hours = max((job["end"] - job["start"]).total_seconds(), 0) / 3600
        job["outcome"] = job_outcome(job)
        job["hours"] = hours
        log_file = job["path"]
        if log_file.is_relative_to(REPO_DIR):
            log_file = log_file.relative_to(REPO_DIR)
        job_rows.append({
            "experiment": experiment,
            "job_id": cell(job["job_id"]),
            "start": cell(job["start"] and job["start"].isoformat(sep=" ")),
            "end": cell(job["end"] and job["end"].isoformat(sep=" ")),
            "hours": cell(hours, ".2f"),
            "nodes": cell(job["nodes"]),
            "gpus": cell(gpus),
            "gpu_hours": cell(hours and hours * gpus, ".2f"),
            "resubmission": cell(job["resubmission"]),
            "resumed_from_step": cell(job["resumed_from_step"]),
            "stopped_at_step": cell(job["stopped_at_step"]),
            "outcome": job["outcome"],
            "peak_gpu_mem_gb": cell(job["peak_gpu_mem_gb"]),
            "gpu_name": cell(job["gpu_name"]),
            "log_file": str(log_file),
        })

    timed = [job for job in jobs if job["hours"] is not None]
    peaks = [job["peak_gpu_mem_gb"] for job in jobs if job["peak_gpu_mem_gb"] is not None]
    if not output_dir.is_dir():
        status = "missing"
    elif final_model_is_complete(final_model):
        status = "complete"
    else:
        status = "incomplete"
    row = {
        "experiment": experiment,
        "status": status,
        "last_outcome": jobs[-1]["outcome"] if jobs else "",
        "model_name": cell(model_name),
        "parameters": cell(parameters),
        "model_gb": cell(sum(path.stat().st_size for path in weights) / 1e9, ".2f") if weights else "",
        "finetune_from": cell(settings.get("finetune_model_path")),
        "dataset": Path(str(data_dir)).name if data_dir else "",
        "train_rows": cell(split_rows.get("train")),
        "validation_rows": cell(split_rows.get("validation")),
        "test_rows": cell(split_rows.get("test")),
        "input_max_length": cell(settings.get("input_max_length")),
        "target_max_length": cell(settings.get("target_max_length")),
        "epochs": cell(settings.get("num_epochs"), "g"),
        "epochs_done": cell(state and round(state.get("epoch") or 0, 2), "g"),
        "steps_done": cell(state and state.get("global_step")),
        "total_steps": cell(state and state.get("max_steps")),
        "per_gpu_batch": cell(per_gpu),
        "grad_accumulation": cell(accumulation),
        "nodes": cell(nodes),
        "gpus": cell(world_size),
        "global_batch": cell(global_batch),
        "parallel_mode": cell(settings.get("parallel_mode")),
        "gradient_checkpointing": cell(
            None if settings.get("gradient_checkpointing") is None
            else bool(settings["gradient_checkpointing"])
        ),
        "learning_rate": cell(
            settings.get("effective_learning_rate", settings.get("learning_rate")), "g"
        ),
        "lr_scaling": cell(settings.get("lr_scaling")),
        "warmup_ratio": cell(settings.get("warmup_ratio"), "g"),
        "weight_decay": cell(settings.get("weight_decay"), "g"),
        "seed": cell(settings.get("seed")),
        "train_loss": cell(train_loss, ".4f"),
        "eval_loss": cell(eval_entries[-1]["eval_loss"] if eval_entries else None, ".4f"),
        "best_eval_loss": cell(best and best["eval_loss"], ".4f"),
        "best_eval_epoch": cell(best and round(best["epoch"], 2), "g"),
        "quick_test_exact_match": cell(quick_test.get("test_exact_match"), ".4f"),
        "quick_test_samples": cell(quick_test.get("test_samples")),
        "jobs": cell(len(jobs)),
        "failed_jobs": cell(sum(job["outcome"] in FAILED_OUTCOMES for job in jobs)),
        "wall_hours": cell(sum(job["hours"] for job in timed), ".2f") if timed else "",
        "gpu_hours": cell(sum(job["hours"] * job["gpus"] for job in timed), ".2f") if timed else "",
        "peak_gpu_mem_gb": cell(max(peaks, default=None)),
        "gpu_name": next((job["gpu_name"] for job in reversed(jobs) if job["gpu_name"]), ""),
        "started": job_rows[0]["start"] if job_rows else "",
        "ended": job_rows[-1]["end"] if job_rows else "",
        "job_ids": ",".join(job["job_id"] for job in jobs if job["job_id"]),
        "config_source": source,
        "output_dir": folder_key(output_dir),
        "date": datetime.date.today().isoformat(),
    }
    return row, job_rows


def training_logs(logs_dir: Path, keys: set[str] | None = None) -> dict[str, list[dict]]:
    """Parsed training logs by the folder_key of their output folder.

    With keys, only the logs of those folders are parsed.
    """
    logs = {}
    if not logs_dir.is_dir():
        return logs
    for path in sorted(logs_dir.glob("*.out")):
        key = log_output_key(path)
        if key and (keys is None or key in keys):
            logs.setdefault(key, []).append(parse_log(path))
    return logs


def is_run_folder(folder: Path) -> bool:
    return folder.is_dir() and (
        any((folder / name).exists() for name in ("run_config.json", "trainer_state.json", "final_model"))
        or any(folder.glob("checkpoint-*"))
    )


def main() -> None:
    args = parse_args()
    if args.all:
        logs = training_logs(args.logs_dir)
        keys = set(logs)
        if args.outputs_dir.is_dir():
            keys.update(
                folder_key(Path("outputs") / folder.name)
                for folder in args.outputs_dir.iterdir()
                if is_run_folder(folder)
            )
        names = configured_runs()
        runs = [
            (names.get(key) or Path(key).name, key_folder(key, args.outputs_dir), logs.get(key, []))
            for key in keys
        ]
    else:
        output_dir = args.output_dir or run_output_dir(args.run)
        key = folder_key(output_dir)
        logs = training_logs(args.logs_dir, {key})
        if not output_dir.is_dir() and key not in logs:
            sys.exit(f"No output folder {output_dir} and no training logs for it in {args.logs_dir}")
        runs = [(args.experiment or experiment_name(args.run, output_dir), output_dir, logs.get(key, []))]

    summaries = [summarise_run(*run) for run in runs]
    # Oldest first, so the rows come out in the order the runs started.
    summaries.sort(key=lambda summary: (summary[0]["started"] or "~", summary[0]["experiment"]))
    for row, job_rows in summaries:
        print(
            f"{row['experiment']}: {row['status']}, epoch {row['epochs_done'] or '?'} "
            f"of {row['epochs'] or '?'}, {row['jobs']} jobs ({row['last_outcome'] or 'no log'}), "
            f"{row['wall_hours'] or '?'} h, {row['gpu_hours'] or '?'} GPU h"
        )
        if not args.no_summary:
            record_rows(args.summary, row["experiment"], [row], SUMMARY_COLUMNS)
            record_rows(args.jobs, row["experiment"], job_rows, JOB_COLUMNS)
    if not args.no_summary:
        print("Recorded in:", args.summary, "and", args.jobs)


if __name__ == "__main__":
    main()
