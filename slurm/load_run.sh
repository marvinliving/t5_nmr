# Load configs/train/$RUN.env and fill in the defaults every job needs.
# Sourced from the repository root by submit.sh and the Slurm jobs.

if [ -z "${RUN:-}" ]; then
  echo "RUN is not set" >&2
  exit 1
fi

RUN_CONFIG="configs/train/$RUN.env"
if [ ! -f "$RUN_CONFIG" ]; then
  echo "No config $RUN_CONFIG; available runs:" >&2
  ls configs/train | sed 's/\.env$//' >&2
  exit 1
fi

# The value a config file sets for a variable, or nothing.
config_value() {
  (unset "$2"; source "$1"; echo "${!2:-}")
}

# The output folder and the model to fine-tune come from the configs or are
# derived below, never from the environment, so a job and the jobs it
# resubmits always agree on them.
unset OUTPUT_DIR FINETUNE_FROM FINETUNE_MODEL_PATH

# A fine-tuning run (FINETUNE_FROM=<run> in its config) starts from that run's
# final model, and its config holds only what differs from that run's: the
# base run's config is loaded first, then the run's own on top.
BASE_RUN=$(config_value "$RUN_CONFIG" FINETUNE_FROM)
if [ -n "$BASE_RUN" ]; then
  base_config="configs/train/$BASE_RUN.env"
  if [ "$BASE_RUN" = "$RUN" ]; then
    echo "$RUN_CONFIG: FINETUNE_FROM names the run itself" >&2
    exit 1
  fi
  if [ ! -f "$base_config" ]; then
    echo "$RUN_CONFIG: FINETUNE_FROM=$BASE_RUN, but there is no $base_config" >&2
    exit 1
  fi
  if [ -n "$(config_value "$base_config" FINETUNE_FROM)" ]; then
    echo "$RUN_CONFIG: $BASE_RUN is itself a fine-tuning run; start from its model with FINETUNE_MODEL_PATH and the settings it needs instead" >&2
    exit 1
  fi
  base_dir=$(config_value "$base_config" OUTPUT_DIR)
  base_dir="${base_dir:-outputs/$BASE_RUN}"
  set -a
  source "$base_config"
  set +a
  # Not inherited: the base run's folder, and the learning rate, length and
  # checkpointing of its training, which take the fine-tuning defaults in
  # config.py unless the run's own config sets them.
  unset OUTPUT_DIR LEARNING_RATE NUM_EPOCHS SAVE_STRATEGY
fi

# Every variable in the config is exported to t5_train.py.
set -a
source "$RUN_CONFIG"
set +a

# The model a fine-tuning run starts from; empty for a run from the
# pretrained MODEL_NAME.
if [ -n "$BASE_RUN" ]; then
  FINETUNE_MODEL_PATH="${FINETUNE_MODEL_PATH:-$base_dir/final_model}"
fi
export FINETUNE_MODEL_PATH="${FINETUNE_MODEL_PATH:-}"

# --nodes and --gpus given to submit.sh (CLI_NODES, CLI_GPUS) override the
# config, so one config can run on any number of nodes.
export NODES="${CLI_NODES:-${NODES:-1}}"
export GPUS_PER_NODE="${CLI_GPUS:-${GPUS_PER_NODE:-1}}"
export DATA_DIR="${DATA_DIR:-alberts_2d}"

# Examples per update: per-GPU batch x accumulation x GPUs.
GLOBAL_BATCH=$(( ${PER_GPU_BATCH:-${TRAIN_BATCH_SIZE:-16}} * ${GRAD_ACCUMULATION_STEPS:-1} * NODES * GPUS_PER_NODE ))

# Output folders are named after the run, except for runs made before this
# layout, whose configs set OUTPUT_DIR to the existing folder. When the GPU
# count comes from the command line, the global batch goes into the name, so
# different node counts never share checkpoints.
if [ -z "${OUTPUT_DIR:-}" ]; then
  OUTPUT_DIR="outputs/$RUN"
  if [ -n "${CLI_NODES:-}${CLI_GPUS:-}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_gb${GLOBAL_BATCH}"
  fi
fi

# A smoke test (MAX_STEPS set at submission) gets its own folder, so it never
# resumes from, or is resumed by, the real run.
if [ "${MAX_STEPS:-0}" -gt 0 ]; then
  OUTPUT_DIR="${OUTPUT_DIR}_max${MAX_STEPS}steps"
fi
export OUTPUT_DIR
# The model evaluate scores.
export MODEL_PATH="$OUTPUT_DIR/final_model"

# Dataset that evaluate, check and progress score on: the run's own unless
# EVAL_DATA_DIR is given, for example an external test set. Results on
# another dataset go to their own folder, so they never overwrite (or get
# combined with) the results on the run's own.
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR}"
if [ "$EVAL_DATA_DIR" = "$DATA_DIR" ]; then
  export EVAL_RESULTS_DIR="$OUTPUT_DIR"
else
  export EVAL_RESULTS_DIR="$OUTPUT_DIR/eval_$(basename "$EVAL_DATA_DIR")"
fi
