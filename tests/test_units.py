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


def write_trainer_state(path: Path, step: int, epoch: float, eval_losses: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    log = [{"step": s, "epoch": s / 10, "eval_loss": loss} for s, loss in eval_losses.items()]
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
