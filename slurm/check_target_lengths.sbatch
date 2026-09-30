#!/bin/bash
# Checks that no reference SMILES in the datasets is longer than training or
# generation allows: ./submit.sh check-lengths. The job fails if one is.
# Needs only the tokenizer, no GPU.
#SBATCH --job-name=check_lengths
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
setup_job_env

# CHECK_DATA_DIRS (slurm/env.sh) lists the datasets; TARGET_MAX_LENGTH and
# GENERATION_MAX_NEW_TOKENS, if set at submission, replace the 640 defaults.
# shellcheck disable=SC2086
python scripts/check_target_lengths.py $CHECK_DATA_DIRS \
  --model-name "${MODEL_NAME:-google/flan-t5-small}" \
  --local-files-only
