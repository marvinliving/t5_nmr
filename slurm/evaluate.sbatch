#!/bin/bash
# Evaluates outputs/<run>/final_model on the full test split as an array job:
# the split is cut into (array tasks x EVAL_GPUS) equal chunks, and each array
# task evaluates EVAL_GPUS of them at once, one per GPU. Submit with
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
EVAL_GPUS="${EVAL_GPUS:-4}"
CHUNKS=$((SLURM_ARRAY_TASK_COUNT * EVAL_GPUS))
CHUNK_SIZE=$(( (TOTAL + CHUNKS - 1) / CHUNKS ))
CHUNK_SIZE=$(( (CHUNK_SIZE + 999) / 1000 * 1000 ))
export TOTAL CHUNK_SIZE NUM_OUTPUTS EVAL_SPLIT

echo "Run: $RUN"
echo "Model: $OUTPUT_DIR/final_model"
echo "Data: $EVAL_DATA_DIR ($EVAL_SPLIT split)"
echo "Results: $EVAL_RESULTS_DIR"
echo "Array task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT, $EVAL_GPUS GPU(s)"
echo "Chunks: $CHUNKS of $CHUNK_SIZE spectra ($TOTAL in all)"
echo "Outputs per spectrum: $NUM_OUTPUTS"

# Evaluates chunk number $1 on the one GPU this process uses.
evaluate_chunk() {
  local chunk="$1"
  local start=$((chunk * CHUNK_SIZE))
  local size="$CHUNK_SIZE"
  # The last chunk stops at TOTAL (which EVAL_MAX_ROWS may have lowered).
  if [ $((start + size)) -gt "$TOTAL" ]; then
    size=$((TOTAL - start))
  fi
  echo "Chunk $chunk: $start + $size of $TOTAL on GPU ${CUDA_VISIBLE_DEVICES:-?}"
  if [ "$start" -ge "$TOTAL" ]; then
    echo "Nothing to evaluate in chunk $chunk"
    return 0
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
    --start-index "$start" \
    --sample-size "$size" \
    --resume \
    --local-files-only
}

if [ "$EVAL_GPUS" -gt 1 ]; then
  # srun starts one task per GPU (see submit.sh), each with its own log,
  # logs/<job name>_<array job>_<array task>_gpu<n>.out.
  export -f evaluate_chunk
  export SLURM_ARRAY_TASK_ID EVAL_GPUS
  srun --cpu-bind=cores \
    --output="logs/%x_%A_%a_gpu%t.out" \
    bash -c '
      # Keep only this task'"'"'s GPU if Slurm shows each task all of them.
      IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-}"
      if [ "${#gpus[@]}" -gt 1 ]; then
        export CUDA_VISIBLE_DEVICES="${gpus[$SLURM_LOCALID]}"
      fi
      evaluate_chunk $((SLURM_ARRAY_TASK_ID * EVAL_GPUS + SLURM_PROCID))
    '
else
  evaluate_chunk "$SLURM_ARRAY_TASK_ID"
fi
