#!/bin/bash
# Evaluates the first 10 test molecules with outputs/<run>/final_model, to
# confirm a model loads and runs. Submit with ./submit.sh check <run>.
#SBATCH --output=logs/%x_%j.out

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh
setup_job_env

nvidia-smi

python evaluate_exact_match.py \
  --model-path "$OUTPUT_DIR/final_model" \
  --data-dir "$EVAL_DATA_DIR" \
  --split test \
  --num-outputs "${NUM_OUTPUTS:-10}" \
  --batch-size "${EVAL_GENERATION_BATCH_SIZE:-16}" \
  --start-index 0 \
  --sample-size 10 \
  --output-file "$EVAL_RESULTS_DIR/test_check.json" \
  --local-files-only
