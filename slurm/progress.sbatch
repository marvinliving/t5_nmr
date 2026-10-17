#!/bin/bash
# Scores exact match for every saved checkpoint of a run and its final
# model, to see whether training has levelled off. Can run while the run
# is still training. Submit with ./submit.sh progress <run>; by default it
# scores greedy top-1 on the first 1,000 test molecules of the external
# dataset (PROGRESS_DATA_DIR in slurm/env.sh), on one node with 4 GPUs,
# each scoring its own share of the checkpoints. For example
#   ./submit.sh progress xxl_4x1x4_10ep --samples 5000 --top 10
#   EVAL_DATA_DIR=/projects/b5an/alberts_2d PROGRESS_SPLIT=validation ./submit.sh progress xxl_4x1x4_10ep
#SBATCH --output=logs/%x_%j.out

set -e
export PYTHONUNBUFFERED=1
# As in evaluate.sbatch: beam search fragments GPU memory without this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh
setup_job_env

# --samples and --top given to submit.sh win over the config.
samples="${CLI_PROGRESS_SAMPLES:-${PROGRESS_SAMPLES:-1000}}"
top="${CLI_PROGRESS_TOP:-${PROGRESS_TOP:-1}}"

# Settings shared by the scoring processes and the table.
options=(
  "$OUTPUT_DIR"
  --base-model "${MODEL_NAME:-google/flan-t5-base}"
  --data-dir "$EVAL_DATA_DIR"
  --split "${PROGRESS_SPLIT:-test}"
  --samples "$samples"
  --num-outputs "$top"
  --batch-size "${EVAL_GENERATION_BATCH_SIZE:-16}"
  --max-batch-tokens "${EVAL_MAX_BATCH_TOKENS:-100000}"
  --max-new-tokens "${GENERATION_MAX_NEW_TOKENS:-640}"
  --local-files-only
)

echo "Run: $RUN"
echo "Checkpoints: $OUTPUT_DIR"
echo "Data: $EVAL_DATA_DIR (${PROGRESS_SPLIT:-test} split), $samples molecules, top-$top"
echo "GPUs: $SLURM_NTASKS"

# One scoring process per GPU, each with its own log,
# logs/progress_<run>_<job>_gpu<n>.out.
status=0
srun --cpu-bind=cores --output="logs/%x_%j_gpu%t.out" bash -c '
  # Keep only this task'"'"'s GPU if Slurm shows each task all of them.
  IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
  if [ "${#gpus[@]}" -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES="${gpus[$SLURM_LOCALID]}"
  fi
  exec python scripts/checkpoint_progress.py "$@" --task "$SLURM_PROCID" --tasks "$SLURM_NTASKS"
' bash "${options[@]}" || status=$?

# The table of every model scored so far, printed even if a GPU failed.
python scripts/checkpoint_progress.py "${options[@]}" --table-only
# The whole loss curve, to outputs/<run>/loss_curve.tsv.
python scripts/loss_curve.py "$OUTPUT_DIR" || echo "No loss curve yet"
exit "$status"
