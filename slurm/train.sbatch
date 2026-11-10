#!/bin/bash
# Trains the model described by configs/train/$RUN.env.
# Submit with ./submit.sh train <run>, which sets the job's nodes, GPUs,
# memory and time from the config.
#SBATCH --output=logs/%x_%j.out

# Exit immediately if a command fails.
set -e

# Make Python output appear in the Slurm log immediately.
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh


# ===== Job time limit =====

# t5_train.py saves a checkpoint and stops STOP_MARGIN_MINUTES before this
# time; the end of this script then resubmits the run.
if [ -z "${SLURM_JOB_END_TIME:-}" ]; then
  SLURM_JOB_END_TIME=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || echo 0)
fi
export SLURM_JOB_END_TIME


# ===== Slurm information =====

echo "===== Slurm Job Info ====="
echo "Run: $RUN ($RUN_CONFIG)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES)"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Global batch: $GLOBAL_BATCH"
echo "Start time: $(date)"
echo "Time limit ends: $(date -d "@$SLURM_JOB_END_TIME" 2>/dev/null || echo unknown)"
echo "Resubmission: ${RESUBMIT_COUNT:-0} of ${MAX_RESUBMITS:-20}"
echo "Output directory: $OUTPUT_DIR"


# ===== Environment activation =====

setup_job_env
echo "Python executable: $(which python)"
python --version
nvidia-smi


# ===== Training summary =====

# However the job ends, record the run in reports/training_summary.tsv and
# its jobs, this one included, in reports/training_jobs.tsv. The exit status
# stays the job's own, and a resubmission is already queued by then.
record_training() {
  python scripts/train_summary.py "$RUN" --output-dir "$OUTPUT_DIR" \
    || echo "reports/training_summary.tsv not updated"
}
trap record_training EXIT


# ===== Start training =====

echo "===== Start Training ====="

if [ $((NODES * GPUS_PER_NODE)) -gt 1 ]; then
  if [ "$NODES" -gt 1 ]; then
    setup_multinode_nccl
  fi
  # srun starts one task per GPU (see submit.sh); each becomes one training
  # process, and the Trainer starts NCCL from these variables.
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  export WORLD_SIZE="$SLURM_NTASKS"
  # Not 72: the CPU work is data loading, and more threads per rank would
  # compete with the dataloader workers.
  export OMP_NUM_THREADS=8
  export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
  export TOKENIZERS_PARALLELISM=false
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

  srun --cpu-bind=cores bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    exec python -u t5_train.py
  '
else
  srun python -u t5_train.py
fi


# ===== Resubmit until finished =====

# Reached only if training exited cleanly: either it finished, or it stopped
# before the time limit and saved a checkpoint to resume from.
if [ -f "$OUTPUT_DIR/final_model/config.json" ]; then
  echo "===== Training finished ====="
elif [ "${RESUBMIT_COUNT:-0}" -lt "${MAX_RESUBMITS:-20}" ]; then
  echo "===== Resubmitting to continue from the last checkpoint ====="
  RESUBMIT_COUNT=$(( ${RESUBMIT_COUNT:-0} + 1 )) \
    ./submit.sh train "$RUN" --dependency=afterany:"$SLURM_JOB_ID"
else
  echo "===== Not finished, but MAX_RESUBMITS=${MAX_RESUBMITS:-20} reached ====="
  exit 1
fi

echo "End time: $(date)"
