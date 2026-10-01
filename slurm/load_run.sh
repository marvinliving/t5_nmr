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

# The output folder comes from the config or is derived below, never from the
# environment, so a job and the jobs it resubmits always agree on it.
unset OUTPUT_DIR

# Every variable in the config is exported to t5_train.py.
set -a
source "$RUN_CONFIG"
set +a

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
