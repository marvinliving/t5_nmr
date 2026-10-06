#!/bin/bash
# Checks a run's combined predictions with RDKit and scores top-n exact match
# counting only valid SMILES (scripts/valid_smiles.py). Submit with
# ./submit.sh valid <run>, after the evaluation's combine job has written
# prd-<split>.txt. The job takes a whole node and uses every core.
#SBATCH --output=logs/%x_%j.out

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
source slurm/load_run.sh
setup_job_env

EVAL_SPLIT="${EVAL_SPLIT:-test}"
if [ "$EVAL_SPLIT" = validation ]; then split_file=val; else split_file=test; fi
predictions="$EVAL_RESULTS_DIR/prd-$split_file.txt"
if [ ! -f "$predictions" ]; then
  echo "No $predictions: evaluate and combine the run first (./submit.sh evaluate $RUN)" >&2
  exit 1
fi

python scripts/valid_smiles.py "$predictions" \
  --data-dir "$EVAL_DATA_DIR" \
  --split "$EVAL_SPLIT" \
  --workers "${SLURM_CPUS_PER_TASK:-0}"
