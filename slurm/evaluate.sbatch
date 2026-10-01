#!/bin/bash
# Evaluates outputs/<run>/final_model on the full test split as an array job:
# task i of n evaluates the i-th of n equal chunks. Submit with
# ./submit.sh evaluate <run>; EVAL_DATA_DIR=<folder> evaluates on another
# dataset, with results in outputs/<run>/eval_<folder name>/.
#SBATCH --output=logs/%x_%A_%a.out

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh
setup_job_env

# Candidate SMILES generated per spectrum, most likely first (1 = top-1 only).
NUM_OUTPUTS="${NUM_OUTPUTS:-10}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

if [ "$EVAL_SPLIT" = validation ]; then split_file=val; else split_file=test; fi
TOTAL=$(wc -l < "$EVAL_DATA_DIR/tgt-$split_file.txt")
# EVAL_MAX_ROWS evaluates only the first rows, e.g. 20000 to compare recipes.
if [ "${EVAL_MAX_ROWS:-0}" -gt 0 ] && [ "$EVAL_MAX_ROWS" -lt "$TOTAL" ]; then
  TOTAL="$EVAL_MAX_ROWS"
fi
# Chunks are rounded up to a multiple of 1,000, so 4 chunks of the
# 79,441-molecule test set start at 0, 20000, 40000 and 60000.
CHUNK_SIZE=$(( (TOTAL + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
CHUNK_SIZE=$(( (CHUNK_SIZE + 999) / 1000 * 1000 ))
START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
# The last chunk stops at TOTAL (which EVAL_MAX_ROWS may have lowered).
if [ $((START_INDEX + CHUNK_SIZE)) -gt "$TOTAL" ]; then
  CHUNK_SIZE=$((TOTAL - START_INDEX))
fi

echo "Run: $RUN"
echo "Model: $OUTPUT_DIR/final_model"
echo "Data: $EVAL_DATA_DIR ($EVAL_SPLIT split)"
echo "Results: $EVAL_RESULTS_DIR"
echo "Array task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Range: $START_INDEX + $CHUNK_SIZE of $TOTAL"
echo "Outputs per spectrum: $NUM_OUTPUTS"

if [ "$START_INDEX" -ge "$TOTAL" ]; then
  echo "Nothing to evaluate in this chunk"
  exit 0
fi

# --resume continues a chunk that was killed from where it stopped.
python evaluate_exact_match.py \
  --model-path "$OUTPUT_DIR/final_model" \
  --data-dir "$EVAL_DATA_DIR" \
  --output-dir "$EVAL_RESULTS_DIR" \
  --split "$EVAL_SPLIT" \
  --num-outputs "$NUM_OUTPUTS" \
  --batch-size "${EVAL_GENERATION_BATCH_SIZE:-16}" \
  --max-new-tokens "${GENERATION_MAX_NEW_TOKENS:-640}" \
  --start-index "$START_INDEX" \
  --sample-size "$CHUNK_SIZE" \
  --resume \
  --local-files-only
