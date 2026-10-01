#!/bin/bash
# Scores top-1 exact match for every saved checkpoint of a run and its final
# model, to see whether it is still improving. Can run while the run is still
# training. Submit with ./submit.sh progress <run>. By default it scores the
# first 2,000 validation molecules of the run's own dataset; for example
#   EVAL_DATA_DIR=/projects/b5an/nmr_expt_data PROGRESS_SPLIT=test PROGRESS_SAMPLES=0
# scores every test molecule of nmr_expt_data.
#SBATCH --output=logs/%x_%j.out

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh
setup_job_env

python scripts/checkpoint_progress.py "$OUTPUT_DIR" \
  --base-model "${MODEL_NAME:-google/flan-t5-base}" \
  --data-dir "$EVAL_DATA_DIR" \
  --split "${PROGRESS_SPLIT:-validation}" \
  --samples "${PROGRESS_SAMPLES:-2000}" \
  --batch-size "${EVAL_GENERATION_BATCH_SIZE:-16}" \
  --max-new-tokens "${GENERATION_MAX_NEW_TOKENS:-640}" \
  --local-files-only
