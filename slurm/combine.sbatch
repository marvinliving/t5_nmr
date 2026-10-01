#!/bin/bash
# Combines the chunks of an evaluation array job and records the score in
# reports/evaluation_summary.tsv. ./submit.sh evaluate <run> submits it to
# start once every chunk has succeeded.
#SBATCH --output=logs/%x_%j.out

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh
setup_job_env

# Runs with --nodes/--gpus or MAX_STEPS get their own folder, outputs/<run>_gb<N>
# or outputs/<run>_max<N>steps, and their own summary row named after it.
experiment="$RUN"
case "$OUTPUT_DIR" in
  "outputs/${RUN}_"*) experiment="${OUTPUT_DIR#outputs/}" ;;
esac
# Results on another dataset (EVAL_DATA_DIR) get their own row too.
if [ "$EVAL_RESULTS_DIR" != "$OUTPUT_DIR" ]; then
  experiment="${experiment}_on_$(basename "$EVAL_DATA_DIR")"
fi

options=(--output-dir "$EVAL_RESULTS_DIR" --data-dir "$EVAL_DATA_DIR" --split "${EVAL_SPLIT:-test}" --experiment "$experiment")
# EVAL_MAX_ROWS evaluated only the first rows, as in evaluate.sbatch.
if [ "${EVAL_SPLIT:-test}" = validation ]; then split_file=val; else split_file=test; fi
TOTAL=$(wc -l < "$EVAL_DATA_DIR/tgt-$split_file.txt")
if [ "${EVAL_MAX_ROWS:-0}" -gt 0 ] && [ "$EVAL_MAX_ROWS" -lt "$TOTAL" ]; then
  options+=(--total "$EVAL_MAX_ROWS")
fi

python scripts/combine_results.py "$RUN" "${options[@]}"
