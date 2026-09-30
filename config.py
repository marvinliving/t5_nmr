"""Training configuration, read from environment variables.

Slurm configs in configs/train/*.env set these variables; any variable left
unset takes the default below. The variable names are the interface: they are
documented in the README.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from nmr_data import DEFAULT_PREFIX


BASE_DIR = Path(__file__).resolve().parent

PARALLEL_MODES = ("none", "ddp", "fsdp", "hsdp")
LR_SCALINGS = ("none", "sqrt", "linear")


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TrainConfig:
    # ===== Experiment identity =====
    model_name: str
    data_dir: Path
    output_dir: Path
    tokenized_cache_dir: Path
    prefix: str
    target_max_length: int
    seed: int

    # ===== Optimisation (keep identical across runs to compare results) =====
    # LEARNING_RATE is the rate for a global batch of base_batch; with
    # lr_scaling it is scaled to the actual global batch (effective_learning_rate).
    learning_rate: float
    lr_scaling: str
    base_batch: int
    warmup_ratio: float
    weight_decay: float
    train_batch_size: int
    grad_accumulation_steps: int
    num_epochs: float
    max_steps: int
    gradient_checkpointing: bool

    # ===== Throughput (no effect on the optimisation) =====
    eval_batch_size: int
    eval_max_samples: int
    dataloader_num_workers: int
    group_by_length: bool

    # ===== Checkpointing and job time limit =====
    save_steps: int
    save_total_limit: int
    job_end_time: float
    stop_margin_minutes: float

    # ===== End-of-training generation test =====
    test_sample_size: int
    generation_batch_size: int
    generation_max_new_tokens: int

    # ===== Distributed training (set by torchrun) =====
    parallel_mode: str
    gpus_per_node: int
    world_size: int
    rank: int
    local_rank: int

    # Set to 1 on offline nodes after the model is downloaded once.
    local_files_only: bool

    @classmethod
    def from_env(cls) -> "TrainConfig":
        model_name = env_str("MODEL_NAME", "google/flan-t5-base")
        output_dir = Path(
            env_str(
                "OUTPUT_DIR",
                str(BASE_DIR / "outputs" / f"{model_name.split('/')[-1]}_nmr"),
            )
        )
        # PER_GPU_BATCH is the newer name; TRAIN_BATCH_SIZE is kept for the
        # configs written before it.
        per_gpu = os.environ.get("PER_GPU_BATCH")
        legacy = os.environ.get("TRAIN_BATCH_SIZE")
        if per_gpu and legacy and int(per_gpu) != int(legacy):
            raise ValueError(
                f"PER_GPU_BATCH={per_gpu} and TRAIN_BATCH_SIZE={legacy} disagree; set one"
            )
        train_batch_size = int(per_gpu or legacy or 16)
        world_size = env_int("WORLD_SIZE", 1)

        # none: one process, one GPU. ddp: every GPU holds the full model.
        # fsdp: parameters, gradients and optimizer state are sharded over
        # all GPUs. hsdp: sharded within each node, replicated across nodes.
        parallel_mode = env_str(
            "PARALLEL_MODE", "ddp" if world_size > 1 else "none"
        ).strip().lower()
        if parallel_mode not in PARALLEL_MODES:
            raise ValueError(
                f"PARALLEL_MODE must be one of {PARALLEL_MODES}, got {parallel_mode!r}"
            )
        if (parallel_mode == "none") != (world_size == 1):
            raise ValueError(
                f"PARALLEL_MODE={parallel_mode} does not match WORLD_SIZE={world_size}: "
                "use PARALLEL_MODE=none with plain python, ddp or fsdp with torchrun."
            )

        lr_scaling = env_str("LR_SCALING", "none").strip().lower()
        if lr_scaling not in LR_SCALINGS:
            raise ValueError(
                f"LR_SCALING must be one of {LR_SCALINGS}, got {lr_scaling!r}"
            )

        return cls(
            model_name=model_name,
            data_dir=Path(env_str("DATA_DIR", str(BASE_DIR / "alberts_2d"))),
            output_dir=output_dir,
            tokenized_cache_dir=Path(
                env_str("TOKENIZED_CACHE_DIR", str(output_dir / "tokenized"))
            ),
            prefix=env_str("PREFIX", DEFAULT_PREFIX),
            target_max_length=env_int("TARGET_MAX_LENGTH", 640),
            seed=env_int("SEED", 42),
            learning_rate=env_float("LEARNING_RATE", 5e-5),
            lr_scaling=lr_scaling,
            # The global batch the earlier results were trained with.
            base_batch=env_int("BASE_BATCH", 16),
            # Share of training spent warming the learning rate up from 0.
            warmup_ratio=env_float("WARMUP_RATIO", 0),
            weight_decay=env_float("WEIGHT_DECAY", 0.01),
            train_batch_size=train_batch_size,
            grad_accumulation_steps=env_int("GRAD_ACCUMULATION_STEPS", 1),
            num_epochs=env_float("NUM_EPOCHS", 3),
            # 0 trains for NUM_EPOCHS; a positive value stops after that many
            # optimizer steps, for smoke tests.
            max_steps=env_int("MAX_STEPS", 0),
            gradient_checkpointing=env_bool("GRADIENT_CHECKPOINTING", False),
            # Evaluation keeps no gradients or optimizer state, so it fits a
            # larger batch than training.
            eval_batch_size=env_int("EVAL_BATCH_SIZE", 4 * train_batch_size),
            # Validation loss is diagnostic only; the first rows are used so
            # the number is comparable across runs. 0 uses the whole split.
            eval_max_samples=env_int("EVAL_MAX_SAMPLES", 5000),
            dataloader_num_workers=env_int("DATALOADER_NUM_WORKERS", 4),
            # Changes the order samples are drawn in, so off by default until
            # validated against a run without it.
            group_by_length=env_bool("GROUP_BY_LENGTH", False),
            save_steps=env_int("SAVE_STEPS", 2000),
            save_total_limit=env_int("SAVE_TOTAL_LIMIT", 2),
            # Unix time the Slurm job is killed at; 0 means no limit.
            job_end_time=env_float("SLURM_JOB_END_TIME", 0),
            stop_margin_minutes=env_float("STOP_MARGIN_MINUTES", 20),
            test_sample_size=env_int("TEST_SAMPLE_SIZE", 1000),
            generation_batch_size=env_int("GENERATION_BATCH_SIZE", 32),
            generation_max_new_tokens=env_int("GENERATION_MAX_NEW_TOKENS", 640),
            parallel_mode=parallel_mode,
            gpus_per_node=env_int("GPUS_PER_NODE", 1),
            world_size=world_size,
            rank=env_int("RANK", 0),
            local_rank=env_int("LOCAL_RANK", 0),
            local_files_only=env_bool("LOCAL_FILES_ONLY", False),
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def global_batch_size(self) -> int:
        # One training process per GPU, so WORLD_SIZE is the number of GPUs.
        return self.train_batch_size * self.grad_accumulation_steps * self.world_size

    @property
    def effective_learning_rate(self) -> float:
        """LEARNING_RATE scaled from base_batch to the actual global batch."""
        ratio = self.global_batch_size / self.base_batch
        if self.lr_scaling == "linear":
            return self.learning_rate * ratio
        if self.lr_scaling == "sqrt":
            return self.learning_rate * math.sqrt(ratio)
        return self.learning_rate

    def to_dict(self) -> dict:
        values = asdict(self)
        values["global_batch_size"] = self.global_batch_size
        values["effective_learning_rate"] = self.effective_learning_rate
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=4, sort_keys=True) + "\n")
