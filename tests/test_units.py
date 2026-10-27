"""Fast tests of the evaluation bookkeeping and training setup; only tiny models are built."""

import json
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "scripts"))

from combine_results import check_tiling, load_chunks, record_summary  # noqa: E402
from evaluate_exact_match import count_match, resume_predictions  # noqa: E402


def test_count_match_counts_first_rank_and_below():
    matches = [0, 0, 0]
    count_match(matches, ["CCN", "CCO", "C"], "CCO")
    assert matches == [0, 1, 1]
    count_match(matches, ["C", "N", "O"], "CCO")
    assert matches == [0, 1, 1]


def test_resume_cuts_back_to_last_whole_spectrum(tmp_path):
    predictions = tmp_path / "predictions.txt"
    # Two outputs per spectrum: spectrum 0 and 1 complete, spectrum 2 has
    # one line and half of another.
    predictions.write_text("CCO\nCC\nN\nCCN\nC#N\nCC(")
    done, matches = resume_predictions(predictions, ["CCO", "CCN", "C#N"], 2)
    assert done == 2
    assert matches == [1, 2]
    assert predictions.read_text() == "CCO\nCC\nN\nCCN\n"


def test_resume_without_file_starts_at_zero(tmp_path):
    assert resume_predictions(tmp_path / "missing.txt", ["C"], 3) == (0, [0, 0, 0])


def test_count_match_ignores_spaces_only_when_asked():
    matches = [0, 0]
    count_match(matches, ["CCN", "CC(=O)O"], "C C ( = O ) O")
    assert matches == [0, 0]
    count_match(matches, ["CCN", "CC(=O)O"], "C C ( = O ) O", ignore_spaces=True)
    assert matches == [0, 1]
    count_match(matches, ["C C N", "C"], "CCN", ignore_spaces=True)
    assert matches == [1, 2]


def test_resume_rescores_without_spaces(tmp_path):
    # A no-space model's predictions against spaced references.
    predictions = tmp_path / "predictions.txt"
    predictions.write_text("CCO\nCCN\n")
    targets = ["C C O", "C C C"]
    assert resume_predictions(predictions, targets, 1) == (2, [0])
    assert resume_predictions(predictions, targets, 1, ignore_spaces=True) == (2, [1])


def test_smiles_spacing_by_majority():
    from nmr_data import smiles_spacing

    assert smiles_spacing(["C C O\n", "C\n", "C C ( = O ) O\n"]) == "spaced"
    # A stray space is outvoted.
    assert smiles_spacing(["CCO", "CC(=O)O", "C C"]) == "nospace"
    assert smiles_spacing(["", "  \n"]) is None


def test_spacing_rule_follows_the_model_and_the_references(tmp_path):
    from nmr_data import model_spacing, spacing_rule

    data_dir = tmp_path / "nospace"
    data_dir.mkdir()
    (data_dir / "tgt-train.txt").write_text("CCO\nCC(=O)O\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({"data_dir": str(data_dir)}))

    model, source = model_spacing(run_dir)
    assert model == "nospace" and str(data_dir) in source
    assert model_spacing(run_dir, "spaced") == ("spaced", "set by SMILES_SPACING")
    assert model_spacing(tmp_path)[0] is None  # no run_config.json

    assert spacing_rule("nospace", source, "spaced")[0] is True
    assert spacing_rule("spaced", source, "spaced")[0] is False
    # Unknown: removing spaces is right either way.
    ignore_spaces, line = spacing_rule(None, "no run_config.json", "spaced")
    assert ignore_spaces and line.endswith("compared without spaces")


def test_combine_refuses_chunks_scored_with_different_spacing_rules():
    from combine_results import check_spacing_rule

    # Results from before the rule was recorded compared as written.
    assert check_spacing_rule([{"start_index": 0}, {"ignore_spaces": False}]) is False
    assert check_spacing_rule([{"ignore_spaces": True}, {"ignore_spaces": True}]) is True
    with pytest.raises(ValueError, match="re-scores"):
        check_spacing_rule([{"ignore_spaces": True}, {}])


def write_chunk(directory: Path, start: int, end: int, matches: list[int]) -> None:
    (directory / f"test_{start}_{end}_results.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "start_index": start,
                "end_index": end,
                "samples": end - start,
                "matches": matches[0],
                "top_n_matches": matches,
            }
        )
    )


def test_chunks_accepted_when_they_tile_the_split(tmp_path):
    write_chunk(tmp_path, 5, 10, [1, 2])
    write_chunk(tmp_path, 0, 5, [3, 4])
    chunks = load_chunks(tmp_path, "test")
    assert [chunk["start_index"] for chunk in chunks] == [0, 5]
    check_tiling(chunks, 10)


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 5)],  # missing the end
        [(0, 5), (4, 10)],  # overlap
        [(0, 4), (5, 10)],  # gap
    ],
)
def test_chunks_rejected_when_they_dont_tile_the_split(tmp_path, ranges):
    for start, end in ranges:
        write_chunk(tmp_path, start, end, [0])
    with pytest.raises(ValueError):
        check_tiling(load_chunks(tmp_path, "test"), 10)


def test_record_summary_replaces_row_of_same_run(tmp_path):
    summary = tmp_path / "summary.tsv"
    row = {"experiment": "xl", "chunks": 4, "total_samples": 10, "exact_match": "0.5"}
    record_summary(summary, row)
    record_summary(summary, {**row, "exact_match": "0.6"})
    lines = summary.read_text().splitlines()
    assert len(lines) == 2
    assert lines[1].split("\t")[3] == "0.6"


def test_target_length_limits_at_their_boundaries():
    from check_target_lengths import rows_over_limits

    # Lengths include end-of-sequence: training keeps 128 tokens with it,
    # generation needs 128 tokens without it.
    truncated, unfinishable = rows_over_limits([128, 129, 130], 128, 128)
    assert truncated == [1, 2]
    assert unfinishable == [2]


def test_split_size_read_from_results(tmp_path):
    import argparse

    from combine_results import split_total

    write_chunk(tmp_path, 0, 10, [1])
    chunks = load_chunks(tmp_path, "test")
    for chunk in chunks:
        chunk["split_size"] = 10
    args = argparse.Namespace(total=None, data_dir=None, run="unused", split="test")
    assert split_total(args, chunks) == 10


def make_config(monkeypatch, **env):
    from config import TrainConfig

    for name in ("WORLD_SIZE", "PARALLEL_MODE", "TRAIN_BATCH_SIZE", "PER_GPU_BATCH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    return TrainConfig.from_env()


@pytest.mark.parametrize(
    "scaling, expected",
    [("none", 5e-5), ("sqrt", 5e-5 * 4), ("linear", 5e-5 * 16)],
)
def test_learning_rate_scales_with_global_batch(monkeypatch, scaling, expected):
    # 4 per GPU x 64 GPUs = global batch 256, 16x the base batch.
    config = make_config(
        monkeypatch, PER_GPU_BATCH=4, WORLD_SIZE=64, PARALLEL_MODE="ddp", LR_SCALING=scaling
    )
    assert config.global_batch_size == 256
    assert config.effective_learning_rate == pytest.approx(expected)


def test_resume_refused_with_different_gpu_count(monkeypatch, tmp_path):
    from t5_train import check_resume_compatible

    written = make_config(
        monkeypatch, OUTPUT_DIR=tmp_path, PER_GPU_BATCH=4, WORLD_SIZE=8, PARALLEL_MODE="ddp"
    )
    written.write_json(tmp_path / "run_config.json")

    same = make_config(
        monkeypatch, OUTPUT_DIR=tmp_path, PER_GPU_BATCH=4, WORLD_SIZE=8, PARALLEL_MODE="ddp"
    )
    check_resume_compatible(same, str(tmp_path / "checkpoint-10"))

    fewer = make_config(
        monkeypatch, OUTPUT_DIR=tmp_path, PER_GPU_BATCH=4, WORLD_SIZE=4, PARALLEL_MODE="ddp"
    )
    with pytest.raises(SystemExit):
        check_resume_compatible(fewer, str(tmp_path / "checkpoint-10"))


@pytest.fixture
def tiny_flan_t5(tmp_path):
    """A tiny checkpoint shaped like FLAN-T5: output layer separate from the input embeddings."""
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    config = T5Config(
        vocab_size=64, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=1,
        feed_forward_proj="gated-gelu", tie_word_embeddings=False,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.lm_head.weight = torch.nn.Parameter(torch.randn_like(model.shared.weight))
    model.save_pretrained(tmp_path)
    return tmp_path


def load_tiny(path, zero_weights=False):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    if zero_weights:
        # What ranks other than local rank 0 see under FSDP with
        # cpu_ram_efficient_loading: all-zero weights, then the load-time tie.
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(all_submodels=True)
        model.tie_weights(missing_keys=set(), recompute_mapping=False)
    return model


@pytest.mark.parametrize("zero_weights", [False, True], ids=["rank0", "other_ranks"])
def test_t5_embeddings_survive_fsdp_preparation(tiny_flan_t5, zero_weights):
    import torch

    from t5_train import fix_t5_embeddings

    reference = load_tiny(tiny_flan_t5)
    expected_names = sorted(name for name, _ in reference.named_parameters())
    assert "lm_head.weight" in expected_names

    model = load_tiny(tiny_flan_t5, zero_weights)
    fix_t5_embeddings(model)
    # accelerate's fsdp2_prepare_model moves the model to meta and re-ties.
    model = model.to(torch.device("meta"))
    model.tie_weights()

    assert sorted(name for name, _ in model.named_parameters()) == expected_names
    assert model.lm_head.weight is not model.shared.weight
    # One module owns the input embedding, so FSDP2 puts it in one group.
    assert model.encoder.embed_tokens is model.shared
    assert model.decoder.embed_tokens is model.shared


def test_t5_embedding_fix_keeps_outputs_and_checkpoint(tiny_flan_t5, tmp_path):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    from t5_train import fix_t5_embeddings

    inputs = {"input_ids": torch.tensor([[5, 6, 7, 1]]), "labels": torch.tensor([[8, 9, 1]])}
    reference = load_tiny(tiny_flan_t5).eval()
    model = load_tiny(tiny_flan_t5).eval()
    fix_t5_embeddings(model)
    with torch.no_grad():
        assert torch.equal(model(**inputs).logits, reference(**inputs).logits)

    model.save_pretrained(tmp_path / "saved")
    reloaded = AutoModelForSeq2SeqLM.from_pretrained(tmp_path / "saved").eval()
    assert torch.equal(reloaded.lm_head.weight, reference.lm_head.weight)
    with torch.no_grad():
        assert torch.equal(reloaded(**inputs).logits, reference(**inputs).logits)


def test_save_strategy_read_and_checked(monkeypatch):
    assert make_config(monkeypatch).save_strategy == "steps"
    assert make_config(monkeypatch, SAVE_STRATEGY="Epoch").save_strategy == "epoch"
    with pytest.raises(ValueError, match="SAVE_STRATEGY"):
        make_config(monkeypatch, SAVE_STRATEGY="best")


def test_stop_skips_the_epoch_end_eval_and_save():
    import torch
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from transformers.trainer_callback import DefaultFlowCallback

    from t5_train import StopBeforeTimeLimit

    args = TrainingArguments(output_dir="unused", eval_strategy="epoch", save_strategy="epoch")
    state = TrainerState(global_step=10, epoch=0.5)
    stopper = StopBeforeTimeLimit(deadline=1, device=torch.device("cpu"), check_every=10)
    stopper.on_step_end(args, state, TrainerControl())
    assert stopper.stopped

    # The Trainer runs DefaultFlowCallback before the stopper.
    control = DefaultFlowCallback().on_epoch_end(args, state, TrainerControl())
    stopper.on_epoch_end(args, state, control)
    assert not control.should_evaluate and not control.should_save


def fsdp_checkpoint_worker(rank, model_dir, checkpoint_dir, port):
    """Train one step under FSDP2 and save the model as accelerate does for SHARDED_STATE_DICT."""
    import os

    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from t5_train import fix_t5_embeddings

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2)
    model = load_tiny(model_dir)
    fix_t5_embeddings(model)
    mesh = init_device_mesh("cpu", (2,))
    for block in model.encoder.block + model.decoder.block:
        fully_shard(block, mesh=mesh)
    fully_shard(model.shared, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    batch = {"input_ids": torch.tensor([[5, 6, 7, 1]]), "labels": torch.tensor([[8, 9, 1]])}
    model(**batch).loss.backward()
    optimizer.step()

    sharded = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    dcp.save({"model": sharded}, storage_writer=dcp.FileSystemWriter(
        str(Path(checkpoint_dir) / "pytorch_model_fsdp_0")
    ))
    full = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if rank == 0:
        torch.save(full, Path(checkpoint_dir) / "expected.pt")
    torch.distributed.destroy_process_group()


def test_fsdp_checkpoint_loads_into_one_model(tiny_flan_t5, tmp_path):
    """Run on two CPU processes; takes about 15 seconds."""
    import socket

    import torch

    from checkpoint_progress import load_model

    checkpoint = tmp_path / "checkpoint-1"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(
        fsdp_checkpoint_worker, args=(tiny_flan_t5, checkpoint, port), nprocs=2
    )
    expected = torch.load(checkpoint / "expected.pt")
    pretrained = load_tiny(tiny_flan_t5).state_dict()
    # The training step changed the weights, so a match means they were loaded.
    assert not torch.equal(expected["lm_head.weight"], pretrained["lm_head.weight"])

    cpu = torch.device("cpu")
    model = load_model(checkpoint, str(tiny_flan_t5), torch.float32, cpu, local_files_only=True)
    state = model.state_dict()
    for name, tensor in expected.items():
        assert torch.equal(state[name], tensor), name
    assert model.lm_head.weight is not model.shared.weight
    assert model.encoder.embed_tokens.weight is model.shared.weight

    # bf16, as on a GPU: the stored fp32 weights are converted.
    model = load_model(checkpoint, str(tiny_flan_t5), torch.bfloat16, cpu, local_files_only=True)
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert torch.equal(model.lm_head.weight, expected["lm_head.weight"].to(torch.bfloat16))


def write_trainer_state(
    path: Path, step: int, epoch: float, eval_losses: dict, train_losses: dict | None = None
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    log = [{"step": s, "epoch": s / 10, "loss": loss} for s, loss in (train_losses or {}).items()]
    log += [{"step": s, "epoch": s / 10, "eval_loss": loss} for s, loss in eval_losses.items()]
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": epoch, "log_history": log})
    )


def test_progress_finds_complete_checkpoints_in_step_order(tmp_path):
    from checkpoint_progress import eval_losses, find_models, training_progress

    losses = {10: 0.9, 20: 0.7, 30: 0.6}
    write_trainer_state(tmp_path / "checkpoint-20", 20, 2.0, {10: 0.9, 20: 0.7})
    write_trainer_state(tmp_path / "checkpoint-10", 10, 1.0, {10: 0.9})
    # Still being saved: no trainer_state.json yet.
    (tmp_path / "checkpoint-30").mkdir()
    assert [path.name for path in find_models(tmp_path)] == ["checkpoint-10", "checkpoint-20"]
    assert training_progress(tmp_path / "checkpoint-20", tmp_path, eval_losses(tmp_path)) == {
        "epoch": 2.0, "step": 20, "eval_loss": 0.7,
    }

    # final_model is added when it is newer than the last checkpoint.
    final_model = tmp_path / "final_model"
    final_model.mkdir()
    (final_model / "config.json").write_text("{}")
    (final_model / "model.safetensors").write_text("")
    write_trainer_state(tmp_path, 30, 3.0, losses)
    assert [path.name for path in find_models(tmp_path)][-1] == "final_model"
    assert training_progress(final_model, tmp_path, eval_losses(tmp_path))["eval_loss"] == 0.6

    # With per-epoch saving the last checkpoint is the final model. Saved
    # before the last evaluation, its own history lacks that loss.
    write_trainer_state(tmp_path / "checkpoint-30", 30, 3.0, {10: 0.9, 20: 0.7})
    assert [path.name for path in find_models(tmp_path)][-1] == "checkpoint-30"
    progress = training_progress(tmp_path / "checkpoint-30", tmp_path, eval_losses(tmp_path))
    assert progress["eval_loss"] == 0.6

    # A checkpoint saved mid-epoch has no loss of its own.
    write_trainer_state(tmp_path / "checkpoint-25", 25, 2.5, {10: 0.9, 20: 0.7})
    assert training_progress(tmp_path / "checkpoint-25", tmp_path, {})["eval_loss"] is None


def test_read_history_merges_checkpoint_histories(tmp_path):
    from loss_curve import read_history

    write_trainer_state(tmp_path / "checkpoint-10", 10, 1.0, {10: 0.9}, {5: 2.0, 10: 1.5})
    write_trainer_state(
        tmp_path / "checkpoint-20", 20, 2.0, {10: 0.9}, {5: 2.0, 10: 1.5, 15: 1.2, 20: 1.0}
    )
    # Written at the end of training, after the last evaluation.
    write_trainer_state(tmp_path, 20, 2.0, {20: 0.7})
    # Still being written.
    (tmp_path / "checkpoint-30").mkdir()
    (tmp_path / "checkpoint-30" / "trainer_state.json").write_text('{"global_step": 30, "log_hi')

    history = read_history(tmp_path)
    assert [(entry["step"], "loss" if "loss" in entry else "eval_loss") for entry in history] == [
        (5, "loss"), (10, "eval_loss"), (10, "loss"), (15, "loss"), (20, "eval_loss"), (20, "loss"),
    ]


def test_mean_train_loss_averages_the_epoch():
    from loss_curve import mean_train_loss

    # Epochs are step / 10, as in write_trainer_state.
    losses = {5: 4.0, 10: 3.0, 15: 2.0, 20: 1.0}
    history = [{"step": s, "epoch": s / 10, "loss": loss} for s, loss in losses.items()]
    history.append({"step": 20, "epoch": 2.0, "eval_loss": 9.0})
    assert mean_train_loss(history, 2.0) == pytest.approx(1.5)
    assert mean_train_loss(history, 1.0) == pytest.approx(3.5)
    assert mean_train_loss(history, 5.0) is None


def write_checkpoint(output_dir: Path, step: int, epoch: float, saved: bool = True) -> Path:
    """A checkpoint with model weights and DDP- and FSDP-style optimizer state."""
    path = output_dir / f"checkpoint-{step}"
    (path / "optimizer_0").mkdir(parents=True)
    (path / "optimizer_0" / "shard").write_text("")
    (path / "optimizer.pt").write_text("")
    (path / "model.safetensors").write_text("")
    if saved:
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "epoch": epoch}))
    return path


WHOLE = ["model.safetensors", "optimizer.pt", "optimizer_0", "trainer_state.json"]
SLIM = ["model.safetensors", "trainer_state.json"]


def test_keep_epoch_checkpoints(tmp_path):
    from types import SimpleNamespace

    from transformers import TrainerControl, TrainerState

    from t5_train import KeepEpochCheckpoints

    def files(step):
        return sorted(path.name for path in (tmp_path / f"checkpoint-{step}").iterdir())

    args = SimpleNamespace(output_dir=str(tmp_path))
    keep = KeepEpochCheckpoints()
    for step, epoch in [(100, 1.0), (140, 1.4), (200, 2.0), (270, 2.7)]:
        write_checkpoint(tmp_path, step, epoch)
    keep.on_save(args, TrainerState(), TrainerControl())
    # The time-limit checkpoint is gone once a newer one exists; the newest
    # two are whole; older epochs keep only what scoring needs.
    assert not (tmp_path / "checkpoint-140").exists()
    assert files(200) == WHOLE and files(270) == WHOLE
    assert files(100) == SLIM

    # The next epoch ends: the time-limit checkpoint 270 goes, and so does a
    # checkpoint that a crash left half saved.
    write_checkpoint(tmp_path, 280, 0, saved=False)
    write_checkpoint(tmp_path, 300, 3.0)
    keep.on_save(args, TrainerState(), TrainerControl())
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "checkpoint-100", "checkpoint-200", "checkpoint-300",
    ]
    assert files(200) == WHOLE and files(300) == WHOLE
    assert files(100) == SLIM

    # Only world rank 0 deletes anything.
    write_checkpoint(tmp_path, 350, 3.5)
    write_checkpoint(tmp_path, 400, 4.0)
    keep.on_save(args, TrainerState(is_world_process_zero=False), TrainerControl())
    assert (tmp_path / "checkpoint-350").exists()
    assert files(200) == WHOLE


def test_progress_settings_from_before_top_n_count_as_top1():
    from checkpoint_progress import same_settings

    settings = {
        "data_dir": "/data", "split": "test", "samples": 8, "max_new_tokens": 16,
        "prefix": "p: ", "num_outputs": 1,
    }
    old_result = {key: value for key, value in settings.items() if key != "num_outputs"}
    old_result.update(model="checkpoint-2", exact_match=0.5)
    assert same_settings(old_result, settings)
    assert not same_settings(old_result, {**settings, "num_outputs": 2})


def test_progress_results_from_before_spacing_compared_as_written():
    from checkpoint_progress import same_settings

    settings = {"split": "test", "samples": 8, "ignore_spaces": False}
    old_result = {"split": "test", "samples": 8, "exact_match": 0.0}
    assert same_settings(old_result, settings)
    # A no-space run's old scores are redone with spaces ignored.
    assert not same_settings(old_result, {**settings, "ignore_spaces": True})


def test_logging_steps_read(monkeypatch):
    monkeypatch.delenv("LOGGING_STEPS", raising=False)
    assert make_config(monkeypatch).logging_steps == 500
    assert make_config(monkeypatch, LOGGING_STEPS=1).logging_steps == 1


@pytest.mark.parametrize(
    "eval_data_dir, results_dir",
    [
        (None, "outputs/xl_4x1x4_10ep"),
        ("/projects/b5an/alberts_2d", "outputs/xl_4x1x4_10ep"),
        ("/projects/b5an/nmr_expt_data", "outputs/xl_4x1x4_10ep/eval_nmr_expt_data"),
    ],
)
def test_results_on_another_dataset_get_their_own_folder(eval_data_dir, results_dir):
    import os
    import subprocess

    env = {
        name: value for name, value in os.environ.items()
        if name not in ("DATA_DIR", "EVAL_DATA_DIR", "OUTPUT_DIR", "MAX_STEPS")
    }
    env["RUN"] = "xl_4x1x4_10ep"
    if eval_data_dir:
        env["EVAL_DATA_DIR"] = eval_data_dir
    output = subprocess.run(
        ["bash", "-c", "source slurm/env.sh; source slurm/load_run.sh; echo $EVAL_RESULTS_DIR"],
        cwd=REPO_DIR, env=env, capture_output=True, text=True, check=True,
    ).stdout
    assert output.strip() == results_dir


def loaded(loader: str, env: dict, names: tuple) -> dict:
    """The values of names after sourcing slurm/env.sh and then loader."""
    import os
    import subprocess

    base = {
        name: value for name, value in os.environ.items()
        if name not in ("DATA_DIR", "EVAL_DATA_DIR", "OUTPUT_DIR", "MAX_STEPS", "RUN", "PREDICT")
    }
    script = f"source slurm/env.sh; source {loader}; " + "; ".join(
        f'echo "${{{name}}}"' for name in names
    )
    output = subprocess.run(
        ["bash", "-c", script], cwd=REPO_DIR, env={**base, **env},
        capture_output=True, text=True, check=True,
    ).stdout
    return dict(zip(names, output.splitlines()))


def test_run_model_is_its_final_model():
    values = loaded("slurm/load_run.sh", {"RUN": "xl_4x1x4_10ep"}, ("MODEL_PATH",))
    assert values["MODEL_PATH"] == "outputs/xl_4x1x4_10ep/final_model"


def test_prediction_loads_its_run_and_writes_next_to_the_data():
    names = (
        "RUN", "MODEL_PATH", "MODEL_LABEL", "MODEL_NAME", "EVAL_DATA_DIR",
        "EVAL_RESULTS_DIR", "PREDICTIONS_FILE",
    )
    # A run, dataset and model left in the environment are ignored.
    values = loaded(
        "slurm/load_predict.sh",
        {
            "PREDICT": "xxl_4x1x4_10ep_on_nmr_expt_data",
            "RUN": "xl_4x1x4_10ep",
            "EVAL_DATA_DIR": "/elsewhere",
            "MODEL_PATH": "/elsewhere/model",
        },
        names,
    )
    assert values == {
        "RUN": "xxl_4x1x4_10ep",
        "MODEL_PATH": "outputs/xxl_4x1x4_10ep/final_model",
        "MODEL_LABEL": "xxl_4x1x4_10ep",
        # From the run's config.
        "MODEL_NAME": "google/flan-t5-xxl",
        "EVAL_DATA_DIR": "/projects/b5an/nmr_expt_data",
        "EVAL_RESULTS_DIR": "outputs/predict/xxl_4x1x4_10ep_on_nmr_expt_data",
        "PREDICTIONS_FILE": "/projects/b5an/nmr_expt_data/prd-test_xxl_4x1x4_10ep.txt",
    }


@pytest.fixture
def finetune_configs():
    """Write run configs into configs/train for one test, removed afterwards."""
    written = []

    def write(run: str, text: str) -> str:
        path = REPO_DIR / "configs" / "train" / f"{run}.env"
        written.append(path)
        path.write_text(text)
        return run

    yield write
    for path in written:
        path.unlink(missing_ok=True)


def test_finetune_run_loads_on_top_of_its_base_run(finetune_configs):
    run = finetune_configs(
        "_test_finetune", "FINETUNE_FROM=xxl_4x1x4_10ep\nDATA_DIR=/projects/b5an/other\n"
    )
    names = (
        "MODEL_NAME", "PARALLEL_MODE", "EVAL_BATCH_SIZE", "DATA_DIR", "OUTPUT_DIR",
        "FINETUNE_MODEL_PATH", "LEARNING_RATE", "NUM_EPOCHS", "SAVE_STRATEGY",
    )
    # Values left in the environment, as by an earlier run, are not used either.
    values = loaded(
        "slurm/load_run.sh", {"RUN": run, "LEARNING_RATE": "1", "NUM_EPOCHS": "1"}, names
    )
    assert values == {
        # The base run's settings...
        "MODEL_NAME": "google/flan-t5-xxl",
        "PARALLEL_MODE": "fsdp",
        "EVAL_BATCH_SIZE": "4",
        # ...with the fine-tuning run's own on top.
        "DATA_DIR": "/projects/b5an/other",
        "OUTPUT_DIR": f"outputs/{run}",
        "FINETUNE_MODEL_PATH": "outputs/xxl_4x1x4_10ep/final_model",
        # Not inherited, so config.py's fine-tuning defaults apply.
        "LEARNING_RATE": "",
        "NUM_EPOCHS": "",
        "SAVE_STRATEGY": "",
    }


def test_finetune_run_sets_its_own_learning_rate(finetune_configs):
    run = finetune_configs(
        "_test_finetune", "FINETUNE_FROM=small_4x4_3ep\nLEARNING_RATE=2e-5\nNUM_EPOCHS=5\n"
    )
    names = ("OUTPUT_DIR", "FINETUNE_MODEL_PATH", "LEARNING_RATE", "NUM_EPOCHS")
    assert loaded("slurm/load_run.sh", {"RUN": run}, names) == {
        # The base run's OUTPUT_DIR names the folder of its model, not this run's.
        "OUTPUT_DIR": f"outputs/{run}",
        "FINETUNE_MODEL_PATH": "outputs/flan-t5-small_nmr_input1536_4x4_3ep/final_model",
        "LEARNING_RATE": "2e-5",
        "NUM_EPOCHS": "5",
    }


def test_run_from_the_pretrained_model_ignores_finetuning_in_the_environment():
    values = loaded(
        "slurm/load_run.sh",
        {
            "RUN": "xl_4x1x4_10ep",
            "FINETUNE_FROM": "xxl_4x1x4_10ep",
            "FINETUNE_MODEL_PATH": "/elsewhere/model",
        },
        ("MODEL_NAME", "FINETUNE_MODEL_PATH"),
    )
    assert values == {"MODEL_NAME": "google/flan-t5-xl", "FINETUNE_MODEL_PATH": ""}


def test_finetune_of_a_finetune_refused(finetune_configs):
    import subprocess

    finetune_configs("_test_finetune", "FINETUNE_FROM=xxl_4x1x4_10ep\n")
    run = finetune_configs("_test_finetune_again", "FINETUNE_FROM=_test_finetune\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        loaded("slurm/load_run.sh", {"RUN": run}, ("OUTPUT_DIR",))
    assert "itself a fine-tuning run" in error.value.stderr


def test_finetune_defaults(monkeypatch, tmp_path):
    config = make_config(monkeypatch)
    assert config.finetune_model_path is None
    assert (config.learning_rate, config.save_strategy) == (5e-5, "steps")

    model = tmp_path / "base" / "final_model"
    config = make_config(monkeypatch, FINETUNE_MODEL_PATH=model)
    assert config.finetune_model_path == model
    assert config.to_dict()["finetune_model_path"] == str(model)
    # Smaller steps, and a checkpoint at every epoch to find overtraining.
    assert (config.learning_rate, config.save_strategy) == (1e-5, "epoch")

    config = make_config(
        monkeypatch, FINETUNE_MODEL_PATH=model, LEARNING_RATE=3e-5, SAVE_STRATEGY="steps"
    )
    assert (config.learning_rate, config.save_strategy) == (3e-5, "steps")


def test_finetune_refuses_a_missing_model_and_the_base_runs_folder(monkeypatch, tmp_path):
    from t5_train import check_finetune_model

    model = tmp_path / "base" / "final_model"
    config = make_config(monkeypatch, FINETUNE_MODEL_PATH=model, OUTPUT_DIR=tmp_path / "ft")
    with pytest.raises(SystemExit):
        check_finetune_model(config)

    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text("")
    check_finetune_model(config)

    into_base = make_config(
        monkeypatch, FINETUNE_MODEL_PATH=model, OUTPUT_DIR=tmp_path / "base"
    )
    with pytest.raises(SystemExit):
        check_finetune_model(into_base)


def test_progress_finds_the_model_a_finetune_started_from(tmp_path):
    from checkpoint_progress import start_model

    run = tmp_path / "ft"
    run.mkdir()
    assert start_model(run) is None
    (run / "run_config.json").write_text(json.dumps({"finetune_model_path": None}))
    assert start_model(run) is None

    model = tmp_path / "base" / "final_model"
    (run / "run_config.json").write_text(json.dumps({"finetune_model_path": str(model)}))
    # Deleted since, or never finished.
    assert start_model(run) is None
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text("")
    assert start_model(run) == model
