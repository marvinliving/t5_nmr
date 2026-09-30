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
