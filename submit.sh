#!/bin/bash
# Submits a Slurm job for a run described by configs/train/<run>.env.
#
#   ./submit.sh train <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 train (resubmits itself until done)
#   ./submit.sh evaluate <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 full test-set evaluation (array job),
#                                                 then combine_results.py once it succeeds
#   ./submit.sh check <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 evaluate 10 molecules
#   ./submit.sh progress <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 validation exact match of every
#                                                 checkpoint, to see if it still improves
#   ./submit.sh valid <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 top-n exact match of the evaluated
#                                                 predictions counting only valid SMILES
#   ./submit.sh test-gpu [sbatch options]         GPU and PyTorch check
#   ./submit.sh test-multi-gpu [sbatch options]   NCCL check on 4 GPUs (--nodes=2 for two nodes)
#   ./submit.sh check-lengths [sbatch options]    SMILES longer than the token limits (no GPU)
#
# --nodes and --gpus (GPUs per node) override the run's config; the run's
# output folder then records the global batch, outputs/<run>_gb<N>, and
# evaluate, check and valid need the same values to find it.
#
# Variables not set in the config can be given at submission, for example a
# smoke test: MAX_STEPS=50 ./submit.sh train xxl_2x2x4_10ep

set -e

cd "$(dirname "$0")"
source slurm/env.sh
mkdir -p logs

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 1
}

[ $# -ge 1 ] || usage
job="$1"
shift

case "$job" in
  test-gpu)
    exec sbatch "$@" slurm/test_gpu.sbatch
    ;;
  test-multi-gpu)
    exec sbatch "$@" slurm/test_multi_gpu.sbatch
    ;;
  check-lengths)
    exec sbatch "$@" slurm/check_target_lengths.sbatch
    ;;
  train|evaluate|check|progress|valid)
    [ $# -ge 1 ] || usage
    export RUN="$1"
    shift
    # Take --nodes and --gpus out of the options; the rest go to sbatch.
    sbatch_options=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --nodes=*) export CLI_NODES="${1#*=}" ;;
        --nodes) export CLI_NODES="$2"; shift ;;
        --gpus=*) export CLI_GPUS="${1#*=}" ;;
        --gpus) export CLI_GPUS="$2"; shift ;;
        *) sbatch_options+=("$1") ;;
      esac
      shift
    done
    set -- "${sbatch_options[@]}"
    source slurm/load_run.sh
    ;;
  *)
    usage
    ;;
esac

case "$job" in
  train)
    gpus="$GPUS_PER_NODE"
    if [ $((NODES * gpus)) -gt 1 ]; then
      # Whole-node memory: rank 0 holds the fp32 weights while loading and
      # saving, and every rank memory-maps the tokenized dataset.
      default_mem=0
    else
      default_mem=64G
    fi
    # One task per GH200 superchip, pinned to its own 72-core Grace CPU, as
    # the Isambard-AI documentation recommends.
    exec sbatch \
      --job-name="train_$RUN" \
      --nodes="$NODES" \
      --ntasks-per-node="$gpus" \
      --gpus-per-node="$gpus" \
      --cpus-per-task="$CPUS_PER_GPU" \
      --mem="${TRAIN_MEM:-$default_mem}" \
      --time="${TRAIN_TIME:-1-00:00:00}" \
      --export=ALL \
      "$@" slurm/train.sbatch
    ;;
  evaluate)
    chunks="${EVAL_CHUNKS:-1}"
    # Each array task evaluates EVAL_GPUS chunks at once, one per GPU.
    eval_gpus="${EVAL_GPUS:-4}"
    if [ "$eval_gpus" -gt 1 ]; then default_eval_mem=0; else default_eval_mem=64G; fi
    eval_job=$(sbatch --parsable \
      --job-name="eval_$RUN" \
      --array="0-$((chunks - 1))" \
      --nodes=1 \
      --ntasks-per-node="$eval_gpus" \
      --gpus-per-node="$eval_gpus" \
      --cpus-per-task="$CPUS_PER_GPU" \
      --mem="${EVAL_MEM:-$default_eval_mem}" \
      --time="${EVAL_TIME:-10:00:00}" \
      --export=ALL \
      "$@" slurm/evaluate.sbatch)
    eval_job="${eval_job%%;*}"
    echo "Submitted evaluation array job $eval_job"
    # Combines the chunks once all of them succeed. If a chunk fails, Slurm
    # cancels this job; submitting the evaluation again resumes the chunks
    # and submits a new one.
    sbatch \
      --job-name="combine_$RUN" \
      --dependency="afterok:$eval_job" \
      --kill-on-invalid-dep=yes \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=4 \
      --mem=16G \
      --time=00:30:00 \
      --export=ALL \
      slurm/combine.sbatch
    ;;
  valid)
    # RDKit runs on CPUs only. A whole node bills the same as one core, so
    # take all of it: one worker per core.
    exec sbatch \
      --job-name="valid_$RUN" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task="$((CPUS_PER_GPU * 4))" \
      --mem=0 \
      --time="${VALID_TIME:-02:00:00}" \
      --export=ALL \
      "$@" slurm/valid_smiles.sbatch
    ;;
  check)
    exec sbatch \
      --job-name="check_$RUN" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --gpus-per-node=1 \
      --cpus-per-task=8 \
      --mem=32G \
      --time=00:20:00 \
      --export=ALL \
      "$@" slurm/evaluate_check.sbatch
    ;;
  progress)
    exec sbatch \
      --job-name="progress_$RUN" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --gpus-per-node=1 \
      --cpus-per-task="$CPUS_PER_GPU" \
      --mem="${EVAL_MEM:-64G}" \
      --time="${PROGRESS_TIME:-04:00:00}" \
      --export=ALL \
      "$@" slurm/progress.sbatch
    ;;
esac
