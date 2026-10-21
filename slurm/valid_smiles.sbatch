#!/bin/bash
# Checks a run's combined predictions with RDKit and scores top-n exact match
# counting only valid SMILES (scripts/valid_smiles.py). Submit with
# ./submit.sh valid <run>, after the evaluation's combine job has written
# prd-<split>.txt. ./submit.sh predict <name> submits it after its own
# combine job. The job takes a whole node and uses every core.
#SBATCH --output=logs/%x_%j.out

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
# ./submit.sh predict sets PREDICT: a run's model on the dataset of
# configs/predict/$PREDICT.env.
if [ -n "${PREDICT:-}" ]; then source slurm/load_predict.sh; else source slurm/load_run.sh; fi
setup_job_env

EVAL_SPLIT="${EVAL_SPLIT:-test}"
if [ "$EVAL_SPLIT" = validation ]; then split_file=val; else split_file=test; fi
if [ -n "${PREDICT:-}" ]; then
  predictions="$PREDICTIONS_FILE"
  again="./submit.sh predict $PREDICT"
else
  predictions="$EVAL_RESULTS_DIR/prd-$split_file.txt"
  again="./submit.sh evaluate $RUN"
fi
if [ ! -f "$predictions" ]; then
  echo "No $predictions: evaluate and combine first ($again)" >&2
  exit 1
fi

python scripts/valid_smiles.py "$predictions" \
  --data-dir "$EVAL_DATA_DIR" \
  --split "$EVAL_SPLIT" \
  --smiles-spacing "${SMILES_SPACING:-auto}" \
  --workers "${SLURM_CPUS_PER_TASK:-0}"
