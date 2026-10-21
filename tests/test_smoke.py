"""End-to-end smoke test on CPU: train, evaluate, resume and combine.

Trains google/flan-t5-small for 3 steps on a tiny synthetic dataset, so it
downloads the model on first use (about 300 MB) and takes a few minutes.
It checks the pipeline runs and writes its files, not that the model learns.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "tests"))

from make_tiny_dataset import make_tiny_dataset  # noqa: E402


MODEL_NAME = "google/flan-t5-small"


def run(args, env=None):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_DIR,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    data_dir = make_tiny_dataset(root / "data")
    output_dir = root / "run"
    env = {
        "MODEL_NAME": MODEL_NAME,
        "DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(output_dir),
        "MAX_STEPS": "3",
        "SAVE_STEPS": "2",
        "TRAIN_BATCH_SIZE": "4",
        "GRAD_ACCUMULATION_STEPS": "1",
        "EVAL_MAX_SAMPLES": "16",
        # A training-loss entry at every step, for the loss curve.
        "LOGGING_STEPS": "1",
        "TEST_SAMPLE_SIZE": "8",
        "GENERATION_BATCH_SIZE": "4",
        "GENERATION_MAX_NEW_TOKENS": "16",
        "DATALOADER_NUM_WORKERS": "0",
        "PARALLEL_MODE": "none",
    }
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "SLURM_JOB_END_TIME"):
        os.environ.pop(name, None)
    run(["t5_train.py"], env)
    return data_dir, output_dir, env


def test_training_writes_model_and_metrics(trained):
    _, output_dir, _ = trained
    assert (output_dir / "final_model" / "config.json").is_file()
    assert (output_dir / "checkpoint-2").is_dir()
    assert (output_dir / "tokenized" / "t5_train_fingerprint.json").is_file()

    metrics = json.loads((output_dir / "test_results.json").read_text())
    assert set(metrics) == {"test_exact_match", "test_samples"}
    assert metrics["test_samples"] == 8

    config = json.loads((output_dir / "run_config.json").read_text())
    assert config["max_steps"] == 3
    assert config["global_batch_size"] == 4


def test_finished_run_is_not_retrained(trained):
    _, output_dir, env = trained
    stdout = run(["t5_train.py"], env)
    assert "Training already finished" in stdout


def test_evaluate_resume_and_combine(trained):
    data_dir, output_dir, _ = trained
    model = str(output_dir / "final_model")
    common = [
        "--model-path", model, "--data-dir", str(data_dir), "--split", "test",
        "--num-outputs", "2", "--batch-size", "4", "--max-new-tokens", "16",
        "--save-every", "2", "--sample-size", "32",
    ]
    for start in ("0", "32"):
        run(["evaluate_exact_match.py", *common, "--start-index", start])

    first = json.loads((output_dir / "test_0_32_results.json").read_text())
    predictions = output_dir / "test_0_32_predictions.txt"
    assert len(predictions.read_text().splitlines()) == 64

    # Drop the last spectrum and a half line, then resume.
    lines = predictions.read_text().splitlines(keepends=True)
    predictions.write_text("".join(lines[:-3]) + "CC")
    run(["evaluate_exact_match.py", *common, "--start-index", "0", "--resume"])
    resumed = json.loads((output_dir / "test_0_32_results.json").read_text())
    assert resumed["resumed_samples"] == 30
    assert resumed["samples"] == 32
    assert len(predictions.read_text().splitlines()) == 64

    # Only the last 2 spectra were generated again, in a different batch.
    assert abs(resumed["top_n_matches"][0] - first["top_n_matches"][0]) <= 2

    stdout = run(
        [
            "scripts/combine_results.py", "tiny", "--output-dir", str(output_dir),
            "--data-dir", str(data_dir), "--no-summary",
        ]
    )
    assert "top-2:" in stdout
    assert len((output_dir / "prd-test.txt").read_text().splitlines()) == 128


def test_checkpoint_progress_scores_each_checkpoint(trained):
    data_dir, output_dir, _ = trained
    common = [
        "scripts/checkpoint_progress.py", str(output_dir), "--base-model", MODEL_NAME,
        "--batch-size", "4", "--max-new-tokens", "16",
    ]
    args = [*common, "--data-dir", str(data_dir), "--samples", "8"]
    stdout = run(args)
    assert "Exact match on 8 validation molecules" in stdout
    progress_dir = output_dir / "progress" / "data_validation"
    rows = (progress_dir / "progress.tsv").read_text().splitlines()
    # The Trainer also saves at the last step, so final_model, which holds
    # the same weights as checkpoint-3, is not scored again.
    assert [row.split("\t")[0] for row in rows] == ["model", "checkpoint-2", "checkpoint-3"]
    assert len((progress_dir / "checkpoint-3_predictions.txt").read_text().splitlines()) == 8

    # Scores are kept, so a second run only prints the table.
    stdout = run(args)
    assert stdout.count("already scored") == 2


def test_checkpoint_progress_on_all_of_an_external_test_set(trained, tmp_path):
    _, output_dir, _ = trained
    external = make_tiny_dataset(tmp_path / "external", rows=24)
    args = [
        "scripts/checkpoint_progress.py", str(output_dir), "--base-model", MODEL_NAME,
        "--batch-size", "4", "--max-new-tokens", "16",
        "--data-dir", str(external), "--split", "test", "--samples", "0",
    ]
    stdout = run(args)
    assert "Exact match on 24 test molecules" in stdout
    progress_dir = output_dir / "progress" / "external_test"
    result = json.loads((progress_dir / "checkpoint-3.json").read_text())
    assert result["samples"] == 24 and result["split"] == "test"

    # A job stopped at its time limit after 10 molecules and half a line.
    predictions = progress_dir / "checkpoint-3_predictions.txt"
    lines = predictions.read_text().splitlines(keepends=True)
    predictions.write_text("".join(lines[:10]) + "CC(")
    settings = {key: result[key] for key in ("data_dir", "split", "samples", "max_new_tokens", "prefix")}
    (progress_dir / "checkpoint-3.partial.json").write_text(json.dumps(settings))
    (progress_dir / "checkpoint-3.json").unlink()
    stdout = run(args)
    assert "Resuming after 10 of 24 molecules" in stdout
    assert len(predictions.read_text().splitlines()) == 24
    assert not (progress_dir / "checkpoint-3.partial.json").exists()
    assert json.loads((progress_dir / "checkpoint-3.json").read_text())["samples"] == 24


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def test_checkpoint_progress_shares_models_between_tasks(trained, tmp_path):
    _, output_dir, _ = trained
    shared = make_tiny_dataset(tmp_path / "shared", rows=8)
    args = [
        "scripts/checkpoint_progress.py", str(output_dir), "--base-model", MODEL_NAME,
        "--batch-size", "4", "--max-new-tokens", "16",
        "--data-dir", str(shared), "--split", "test", "--samples", "8",
    ]
    progress_dir = output_dir / "progress" / "shared_test"

    def scored():
        return sorted(path.name for path in progress_dir.glob("checkpoint-*.json"))

    # In the job the tasks list the models at the same time. Run one after
    # the other, task 1 goes first, so task 0 still finds both models to do.
    stdout = run([*args, "--task", "1", "--tasks", "2"])
    assert "Task 1 of 2: scoring checkpoint-3" in stdout
    assert scored() == ["checkpoint-3.json"]
    stdout = run([*args, "--task", "0", "--tasks", "2"])
    assert "Task 0 of 2: scoring checkpoint-2" in stdout
    assert scored() == ["checkpoint-2.json", "checkpoint-3.json"]

    stdout = run([*args, "--table-only"])
    assert "scoring" not in stdout
    assert "Exact match on 8 test molecules" in stdout
    assert "checkpoint-2\t" in stdout and "checkpoint-3\t" in stdout
    rows = read_tsv(progress_dir / "progress.tsv")
    assert [row["model"] for row in rows] == ["checkpoint-2", "checkpoint-3"]
    assert all(row["train_loss"] for row in rows)


def test_checkpoint_progress_top_n(trained):
    data_dir, output_dir, _ = trained
    stdout = run([
        "scripts/checkpoint_progress.py", str(output_dir), "--base-model", MODEL_NAME,
        "--batch-size", "4", "--max-new-tokens", "16",
        "--data-dir", str(data_dir), "--samples", "8", "--num-outputs", "2",
    ])
    assert "Top-1 to top-2 exact match (beam search) on 8 validation molecules" in stdout
    progress_dir = output_dir / "progress" / "data_validation_top2"
    rows = read_tsv(progress_dir / "progress.tsv")
    assert [row["model"] for row in rows] == ["checkpoint-2", "checkpoint-3"]
    assert all(float(row["top_2"]) >= float(row["exact_match"]) for row in rows)
    assert len((progress_dir / "checkpoint-3_predictions.txt").read_text().splitlines()) == 16


def test_loss_curve_written(trained):
    _, output_dir, _ = trained
    stdout = run(["scripts/loss_curve.py", str(output_dir)])
    assert "Saved to:" in stdout
    rows = read_tsv(output_dir / "loss_curve.tsv")
    assert [row["step"] for row in rows] == ["1", "2", "3"]
    assert all(row["loss"] for row in rows)


def test_evaluation_results_go_to_output_dir(trained, tmp_path):
    data_dir, output_dir, _ = trained
    results_dir = tmp_path / "eval_external"
    run([
        "evaluate_exact_match.py", "--model-path", str(output_dir / "final_model"),
        "--data-dir", str(data_dir), "--split", "test", "--batch-size", "4",
        "--max-new-tokens", "16", "--sample-size", "4", "--output-dir", str(results_dir),
    ])
    assert (results_dir / "test_0_4_results.json").is_file()
    assert (results_dir / "test_0_4_predictions.txt").is_file()


def test_prediction_on_a_folder_with_only_a_test_split(trained, tmp_path):
    # The steps of ./submit.sh predict: evaluate, combine next to the spectra, valid.
    data_dir, output_dir, _ = trained
    folder = tmp_path / "test_only"
    folder.mkdir()
    for name in ("src-test.txt", "tgt-test.txt"):
        (folder / name).write_text((data_dir / name).read_text())
    work = tmp_path / "work"
    run([
        "evaluate_exact_match.py", "--model-path", str(output_dir / "final_model"),
        "--data-dir", str(folder), "--split", "test", "--num-outputs", "2",
        "--batch-size", "4", "--max-new-tokens", "16", "--output-dir", str(work),
    ])
    predictions = folder / "prd-test_tiny.txt"
    run([
        "scripts/combine_results.py", "tiny", "--output-dir", str(work),
        "--data-dir", str(folder), "--predictions", str(predictions), "--no-summary",
    ])
    assert len(predictions.read_text().splitlines()) == 2 * 64
    assert not (work / "prd-test.txt").exists()

    pytest.importorskip("rdkit")
    run(["scripts/valid_smiles.py", str(predictions), "--data-dir", str(folder), "--workers", "1"])
    assert json.loads((folder / "prd-test_tiny_valid.json").read_text())["samples"] == 64


def spaced_copy(data_dir: Path, destination: Path) -> Path:
    """The dataset with its SMILES as space-separated characters, as alberts_2d stores them."""
    destination.mkdir(parents=True)
    for path in data_dir.iterdir():
        text = path.read_text()
        if path.name.startswith("tgt-"):
            text = "".join(" ".join(line) + "\n" for line in text.splitlines())
        (destination / path.name).write_text(text)
    return destination


def test_no_space_model_scored_on_spaced_references(trained, tmp_path):
    # The tiny dataset's SMILES have no spaces, so the model writes none.
    data_dir, output_dir, _ = trained
    spaced = spaced_copy(data_dir, tmp_path / "spaced")
    results_dir = tmp_path / "results"
    args = [
        "evaluate_exact_match.py", "--model-path", str(output_dir / "final_model"),
        "--data-dir", str(spaced), "--split", "test", "--batch-size", "4",
        "--max-new-tokens", "16", "--sample-size", "8", "--output-dir", str(results_dir),
    ]
    stdout = run(args)
    assert "SMILES: model nospace (from the training targets in" in stdout
    assert "references spaced: compared without spaces" in stdout

    # Give it the right answers, without spaces, and score them again.
    targets = (data_dir / "tgt-test.txt").read_text().splitlines()[:8]
    (results_dir / "test_0_8_predictions.txt").write_text("".join(t + "\n" for t in targets))
    run([*args, "--resume"])
    result = json.loads((results_dir / "test_0_8_results.json").read_text())
    assert result["model_spacing"] == "nospace" and result["reference_spacing"] == "spaced"
    assert result["ignore_spaces"] is True
    assert result["exact_match"] == 1.0

    # Progress follows the same rule and keeps these scores apart.
    stdout = run([
        "scripts/checkpoint_progress.py", str(output_dir), "--base-model", MODEL_NAME,
        "--batch-size", "4", "--max-new-tokens", "16",
        "--data-dir", str(spaced), "--split", "test", "--samples", "8",
    ])
    assert "spaces ignored =====" in stdout
    progress_dir = output_dir / "progress" / "spaced_test"
    assert json.loads((progress_dir / "checkpoint-3.json").read_text())["ignore_spaces"] is True
