"""Fine-tune FLAN-T5 to predict SMILES from NMR spectra.

Configured through environment variables (see config.py and the README).
Run with plain `python t5_train.py` on one GPU. On several GPUs, slurm/train.sbatch
starts one process per GPU with srun and sets RANK, LOCAL_RANK and WORLD_SIZE;
PARALLEL_MODE chooses ddp, fsdp or hsdp.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from config import TrainConfig
from evaluate_exact_match import run_evaluation
from nmr_data import load_split, load_tokenized_splits


def fix_t5_embeddings(model: torch.nn.Module) -> None:
    """Give every rank FLAN-T5's real embedding structure, which FSDP needs.

    FLAN-T5 has one input embedding (model.shared) used by the encoder and
    the decoder, and a separate output layer (lm_head). transformers 5.12
    gets both wrong for FSDP:

    - It always tells T5 to tie lm_head to the input embeddings, and only
      skips it at load time when the checkpoint holds two different tensors.
      Under FSDP with cpu_ram_efficient_loading, ranks other than local rank
      0 load all-zero weights, so the check passes and those ranks tie
      lm_head while rank 0 does not. The ranks then hold different parameter
      sets and saving the optimizer state fails. FSDP preparation also calls
      tie_weights() again. Removing lm_head from the tie mapping fixes both;
      setting tie_word_embeddings=False instead would also untie the encoder
      and decoder input embeddings.
    - The encoder and decoder each get their own embedding module that shares
      model.shared's weight. accelerate shards model.shared as its own FSDP
      group and the other two modules land in the root group, so the first
      forward fails with "Parameter 'shared.weight' is shared with a
      parameter already managed by another FSDP group". Pointing both at
      model.shared itself, as transformers 4.x did, leaves one owner.

    Neither change alters the computation. Other ranks' lm_head values are
    replaced by rank 0's broadcast.
    """
    model._tied_weights_keys = {
        target: source
        for target, source in model._tied_weights_keys.items()
        if target != "lm_head.weight"
    }
    model.all_tied_weights_keys.pop("lm_head.weight", None)
    if model.lm_head.weight is model.shared.weight:
        model.lm_head.weight = torch.nn.Parameter(model.shared.weight.detach().clone())
    model.set_input_embeddings(model.shared)


class StopBeforeTimeLimit(TrainerCallback):
    """Save a checkpoint and stop cleanly before Slurm kills the job.

    Stops when the deadline (the job's time limit minus a margin) has passed,
    or when the process receives SIGUSR1 (scancel --signal=USR1 <jobid>).
    Every rank must stop at the same step, or the others would wait forever
    in the next collective, so the ranks agree on the decision with an
    all-reduce. It runs every check_every steps to keep its cost negligible.
    """

    def __init__(self, deadline: float, device: torch.device, check_every: int = 10):
        self.deadline = deadline
        self.device = device
        self.check_every = check_every
        self.signalled = False
        self.stopped = False
        signal.signal(signal.SIGUSR1, self.on_signal)

    def on_signal(self, signum, frame):
        self.signalled = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_every:
            return
        due = self.signalled or (self.deadline > 0 and time.time() >= self.deadline)
        stop = torch.tensor([float(due)], device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
        if stop.item():
            self.stopped = True
            control.should_save = True
            control.should_training_stop = True


def save_metrics_file(output_dir: Path, split: str, metrics: dict) -> None:
    """Write <split>_results.json and all_results.json like Trainer.save_metrics."""
    with (output_dir / f"{split}_results.json").open("w") as file:
        json.dump(metrics, file, indent=4, sort_keys=True)

    all_results_path = output_dir / "all_results.json"
    all_metrics = {}
    if all_results_path.is_file():
        with all_results_path.open() as file:
            all_metrics = json.load(file)
    all_metrics.update(metrics)
    with all_results_path.open("w") as file:
        json.dump(all_metrics, file, indent=4, sort_keys=True)


# Settings a checkpoint depends on: resuming with different values would
# either fail to load sharded state or silently change the recipe mid-run.
RESUME_KEYS = ("world_size", "parallel_mode", "global_batch_size", "effective_learning_rate")


def check_resume_compatible(config: TrainConfig, checkpoint: str | None) -> None:
    """Refuse to resume a checkpoint written with a different layout or recipe."""
    previous_path = config.output_dir / "run_config.json"
    if not checkpoint or not previous_path.is_file():
        return
    previous = json.loads(previous_path.read_text())
    current = config.to_dict()
    mismatches = [
        f"  {key}: checkpoint {previous[key]}, now {current[key]}"
        for key in RESUME_KEYS
        if key in previous and previous[key] != current[key]
    ]
    if mismatches:
        if config.is_main_process:
            print(
                f"Cannot resume {checkpoint}: it was written with different settings.\n"
                + "\n".join(mismatches)
                + "\nSubmit with the original number of nodes and GPUs, or use a "
                "new run name to start again.",
                file=sys.stderr,
            )
        sys.exit(1)


def final_model_is_complete(final_model_dir: Path) -> bool:
    return (final_model_dir / "config.json").is_file() and any(
        (final_model_dir / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json")
    )


def print_config(config: TrainConfig, use_bf16: bool) -> None:
    print("===== Experiment Configuration =====")
    print("PyTorch:", torch.__version__)
    if torch.cuda.is_available():
        print(f"GPU {config.local_rank}:", torch.cuda.get_device_name(config.local_rank))
    print("bf16:", use_bf16)
    for key, value in config.to_dict().items():
        print(f"{key}: {value}")
    print("====================================")


def build_training_args(config: TrainConfig, use_bf16: bool) -> Seq2SeqTrainingArguments:
    parallel_args = {}

    if config.parallel_mode == "ddp":
        # T5 uses every parameter in each step, so DDP can skip the search.
        parallel_args["ddp_find_unused_parameters"] = False
        # Fewer, larger gradient all-reduces; matters most between nodes.
        parallel_args["ddp_bucket_cap_mb"] = 200

    if config.parallel_mode == "fsdp":
        # Keys as read by transformers 5.12 (TrainingArguments._process_fsdp_args).
        parallel_args["fsdp"] = True
        parallel_args["fsdp_config"] = {
            "version": 2,
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["T5Block"],
            # Only local rank 0 loads the pretrained weights into host RAM;
            # the other ranks receive them by broadcast.
            "cpu_ram_efficient_loading": True,
            # Checkpoints hold one shard per rank; the final model is
            # gathered into a single file in main().
            "state_dict_type": "SHARDED_STATE_DICT",
        }

    if config.parallel_mode == "hsdp":
        # Hybrid sharding: the model is sharded over the GPUs of each node
        # (NVLink) and replicated across nodes, so only a gradient all-reduce
        # crosses the network. transformers 5.12 offers it through FSDP1.
        parallel_args["fsdp"] = True
        parallel_args["fsdp_config"] = {
            "version": 1,
            "reshard_after_forward": "hybrid_shard",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["T5Block"],
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
            "use_orig_params": True,
            "state_dict_type": "SHARDED_STATE_DICT",
        }

    if config.group_by_length:
        parallel_args["train_sampling_strategy"] = "group_by_length"

    return Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        eval_strategy="epoch",
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        logging_strategy="steps",
        logging_steps=500,
        learning_rate=config.effective_learning_rate,
        # A value below 1 is a share of the total training steps.
        warmup_steps=config.warmup_ratio,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.grad_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        predict_with_generate=False,
        # T5 overflows in fp16, so without bf16 support train in fp32.
        bf16=use_bf16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
        # Rank 0 tokenises the dataset and gathers the full XXL state dict
        # while the other ranks wait; both can exceed the 30-minute default.
        ddp_timeout=7200,
        # The Trainer seeds Python, NumPy and PyTorch from these.
        seed=config.seed,
        data_seed=config.seed,
        **parallel_args,
    )


def main() -> None:
    # With one Slurm task per GPU, a task may see all of the node's GPUs (then
    # LOCAL_RANK picks one) or only its own (then it is device 0).
    if torch.cuda.device_count() == 1:
        os.environ["LOCAL_RANK"] = "0"
    config = TrainConfig.from_env()
    final_model_dir = config.output_dir / "final_model"

    # Resubmitting a finished run would otherwise resume from its last
    # checkpoint, retrain the tail and overwrite final_model.
    if final_model_is_complete(final_model_dir):
        if config.is_main_process:
            print("Training already finished:", final_model_dir)
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(config.output_dir))
    check_resume_compatible(config, checkpoint)

    if config.parallel_mode == "hsdp" and torch.cuda.device_count() != config.gpus_per_node:
        # Hybrid sharding groups the GPUs a process can see into one shard
        # group; seeing only its own GPU would replicate the full model.
        raise RuntimeError(
            f"hsdp needs every process to see all {config.gpus_per_node} GPUs of its "
            f"node, but it sees {torch.cuda.device_count()}"
        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = torch.device(
        f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu"
    )

    # Built first: it starts the distributed process group used below.
    training_args = build_training_args(config, use_bf16)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        local_files_only=config.local_files_only,
    )
    tokenized = load_tokenized_splits(
        config.data_dir,
        tokenizer,
        model_name=config.model_name,
        prefix=config.prefix,
        target_max_length=config.target_max_length,
        cache_dir=config.tokenized_cache_dir,
        num_proc=config.dataloader_num_workers,
        is_main_process=config.is_main_process,
        main_process_first=lambda: training_args.main_process_first(
            local=False, desc="dataset tokenization"
        ),
    )
    eval_dataset = tokenized["validation"]
    if 0 < config.eval_max_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(config.eval_max_samples))

    if config.is_main_process:
        print_config(config, use_bf16)
        print("Train rows:", len(tokenized["train"]))
        print("Validation rows used for eval_loss:", len(eval_dataset))
        config.write_json(config.output_dir / "run_config.json")

    if config.parallel_mode in ("fsdp", "hsdp"):
        # Lets from_pretrained see FSDP before the Trainer creates it, so
        # cpu_ram_efficient_loading applies to this load.
        os.environ["ACCELERATE_USE_FSDP"] = "true"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model_name,
        local_files_only=config.local_files_only,
    )
    fix_t5_embeddings(model)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # Tensor-core friendly shapes; padding is masked, so the loss is unchanged.
        pad_to_multiple_of=8,
    )

    # Without a Slurm time limit (job_end_time 0) it only reacts to SIGUSR1.
    deadline = config.job_end_time - 60 * config.stop_margin_minutes
    stopper = StopBeforeTimeLimit(deadline if config.job_end_time > 0 else 0, device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[stopper],
    )

    if config.is_main_process:
        if checkpoint:
            print("Resuming from checkpoint:", checkpoint)
        else:
            print("Starting training from the pretrained model")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    if stopper.stopped:
        if config.is_main_process:
            print(
                "Stopped before the job time limit at step",
                trainer.state.global_step,
                "- checkpoint saved; submit again to continue.",
            )
        return

    if config.is_main_process:
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if trainer.is_fsdp_enabled:
        # Gather the shards into one Hugging Face model that loads anywhere.
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Called on every rank: under FSDP all ranks take part in the gather.
    trainer.save_model(str(final_model_dir))
    if config.is_main_process:
        tokenizer.save_pretrained(str(final_model_dir))
        print("Final model saved to:", final_model_dir)
        if torch.cuda.is_available():
            print(
                "Peak GPU memory allocated (GB):",
                round(torch.cuda.max_memory_allocated(config.local_rank) / 1e9, 1),
            )

    if config.parallel_mode == "none":
        # The single-GPU run tests the fp32 model it just trained.
        test_model = model
    else:
        # A DDP- or FSDP-wrapped model cannot generate on one rank, so rank 0
        # reloads the saved model in bf16, as evaluate_exact_match.py does.
        torch.distributed.barrier()
        del trainer, model
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        if not config.is_main_process:
            return
        test_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(final_model_dir),
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
        ).to(device)

    # Quick check on the first test rows; the official score comes from
    # evaluate_exact_match.py on the whole split.
    print("===== Bounded Generation Test =====")
    test_dataset = load_split(config.data_dir, "test")
    test_dataset = test_dataset.select(
        range(min(config.test_sample_size, len(test_dataset)))
    )
    test_metrics = {"test_exact_match": 0.0, "test_samples": 0}
    if len(test_dataset):
        matches = run_evaluation(
            test_model,
            tokenizer,
            test_dataset,
            device=device,
            prefix=config.prefix,
            batch_size=config.generation_batch_size,
            max_new_tokens=config.generation_max_new_tokens,
            # Dataset order keeps the number comparable with earlier runs.
            sort_by_length=False,
        )
        test_metrics = {
            "test_exact_match": matches[0] / len(test_dataset),
            "test_samples": len(test_dataset),
        }
    save_metrics_file(config.output_dir, "test", test_metrics)
    print(test_metrics)


if __name__ == "__main__":
    main()
