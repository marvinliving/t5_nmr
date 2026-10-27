#!/bin/bash
# Submits a Slurm job for a run described by configs/train/<run>.env.
#
#   ./submit.sh train <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 train (resubmits itself until done);
#                                                 a config with FINETUNE_FROM=<run>
#                                                 fine-tunes that run's model
#   ./submit.sh evaluate <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 full test-set evaluation (array job),
#                                                 then combine_results.py once it succeeds
#   ./submit.sh check <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 evaluate 10 molecules
#   ./submit.sh progress <run> [--samples N] [--top N] [--nodes N] [--gpus N] [sbatch options]
#                                                 exact match of every saved checkpoint,
#                                                 on one node with 4 GPUs: has training
#                                                 levelled off? Default: greedy top-1 on
#                                                 1000 external test molecules
#   ./submit.sh valid <run> [--nodes N] [--gpus N] [sbatch options]
#                                                 top-n exact match of the evaluated
#                                                 predictions counting only valid SMILES
#   ./submit.sh predict <name> [sbatch options]
#                                                 evaluate, combine and valid for the run
#                                                 and dataset in configs/predict/<name>.env;
#                                                 the predictions go next to its spectra
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
  sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 1
}

# --samples and --top only mean something to the progress job.
progress_only() {
  if [ "$job" != progress ]; then
    echo "$1 is only for ./submit.sh progress" >&2
    exit 1
  fi
}

# How the run's model writes SMILES, for evaluate, check, progress, valid and
# predict; checked at submission so a typo fails there.
check_smiles_spacing() {
  case "${SMILES_SPACING:-auto}" in
    auto|spaced|nospace) ;;
    *) echo "SMILES_SPACING must be auto, spaced or nospace, got $SMILES_SPACING" >&2; exit 1 ;;
  esac
}

# RDKit runs on CPUs only. A whole node bills the same as one core, so take
# all of it: one worker per core.
submit_valid() {
  sbatch \
    --job-name="valid_$label" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task="$((CPUS_PER_GPU * 4))" \
    --mem=0 \
    --time="${VALID_TIME:-02:00:00}" \
    --export=ALL \
    "$@" slurm/valid_smiles.sbatch
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
    # A PREDICT left in the environment would make the jobs load that
    # prediction instead of the run.
    unset PREDICT
    # Take --nodes, --gpus, --samples and --top out of the options; the rest
    # go to sbatch. Like --nodes and --gpus, --samples and --top are kept in
    # CLI_* variables so they win over the run's config.
    sbatch_options=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --nodes=*) export CLI_NODES="${1#*=}" ;;
        --nodes) export CLI_NODES="$2"; shift ;;
        --gpus=*) export CLI_GPUS="${1#*=}" ;;
        --gpus) export CLI_GPUS="$2"; shift ;;
        --samples=*) progress_only "$1"; export CLI_PROGRESS_SAMPLES="${1#*=}" ;;
        --samples) progress_only "$1"; export CLI_PROGRESS_SAMPLES="$2"; shift ;;
        --top=*) progress_only "$1"; export CLI_PROGRESS_TOP="${1#*=}" ;;
        --top) progress_only "$1"; export CLI_PROGRESS_TOP="$2"; shift ;;
        *) sbatch_options+=("$1") ;;
      esac
      shift
    done
    set -- "${sbatch_options[@]}"
    if [ "$job" = progress ]; then
      # progress scores the external test set unless EVAL_DATA_DIR names another.
      export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$PROGRESS_DATA_DIR}"
    fi
    source slurm/load_run.sh
    label="$RUN"
    check_smiles_spacing
    ;;
  predict)
    [ $# -ge 1 ] || usage
    export PREDICT="$1"
    shift
    for option in "$@"; do
      case "$option" in
        --nodes*|--gpus*)
          echo "$option is not for ./submit.sh predict; for a run trained with it, set MODEL_PATH in configs/predict/$PREDICT.env" >&2
          exit 1
          ;;
        --samples*|--top*) progress_only "$option" ;;
      esac
    done
    source slurm/load_predict.sh
    label="predict_$PREDICT"
    # Checked here so a missing model or file fails at submission, not on a
    # GPU node after waiting in the queue.
    case "${EVAL_SPLIT:-test}" in
      test|validation) ;;
      *) echo "EVAL_SPLIT must be test or validation, got $EVAL_SPLIT" >&2; exit 1 ;;
    esac
    if [ ! -f "$MODEL_PATH/config.json" ]; then
      echo "No model in $MODEL_PATH (no config.json): train $RUN first, or set MODEL_PATH in $PREDICT_CONFIG." >&2
      echo "FSDP checkpoints can't be loaded this way; use the run's final_model." >&2
      exit 1
    fi
    if [ "${EVAL_SPLIT:-test}" = validation ]; then split_file=val; else split_file=test; fi
    for file in "src-$split_file.txt" "tgt-$split_file.txt"; do
      if [ ! -f "$EVAL_DATA_DIR/$file" ]; then
        echo "No $EVAL_DATA_DIR/$file" >&2
        exit 1
      fi
    done
    if [ ! -w "$EVAL_DATA_DIR" ]; then
      echo "Can't write the predictions to $EVAL_DATA_DIR" >&2
      exit 1
    fi
    check_smiles_spacing
    echo "Model: $MODEL_PATH"
    echo "Data: $EVAL_DATA_DIR (${EVAL_SPLIT:-test} split)"
    echo "Predictions: $PREDICTIONS_FILE"
    ;;
  *)
    usage
    ;;
esac

case "$job" in
  train)
    # Checked here so a base run that hasn't finished fails at submission,
    # not on a GPU node after waiting in the queue.
    if [ -n "$FINETUNE_MODEL_PATH" ]; then
      if [ ! -f "$FINETUNE_MODEL_PATH/config.json" ]; then
        echo "No model in $FINETUNE_MODEL_PATH (no config.json) to fine-tune: train ${BASE_RUN:-it} first, or set FINETUNE_MODEL_PATH in $RUN_CONFIG." >&2
        exit 1
      fi
      echo "Fine-tuning: $FINETUNE_MODEL_PATH"
    fi
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
  evaluate|predict)
    chunks="${EVAL_CHUNKS:-1}"
    # Each array task evaluates EVAL_GPUS chunks at once, one per GPU.
    eval_gpus="${EVAL_GPUS:-4}"
    if [ "$eval_gpus" -gt 1 ]; then default_eval_mem=0; else default_eval_mem=64G; fi
    eval_job=$(sbatch --parsable \
      --job-name="eval_$label" \
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
    combine_job=$(sbatch --parsable \
      --job-name="combine_$label" \
      --dependency="afterok:$eval_job" \
      --kill-on-invalid-dep=yes \
      --nodes=1 \
      --ntasks-per-node=1 \
      --cpus-per-task=4 \
      --mem=16G \
      --time=00:30:00 \
      --export=ALL \
      slurm/combine.sbatch)
    combine_job="${combine_job%%;*}"
    echo "Submitted combine job $combine_job"
    # A prediction then checks its combined predictions with RDKit; Slurm
    # cancels this too if the combine job fails.
    if [ "$job" = predict ] && [ "${PREDICT_VALID:-1}" != 0 ]; then
      valid_job=$(submit_valid --parsable --dependency="afterok:$combine_job" --kill-on-invalid-dep=yes)
      echo "Submitted valid job ${valid_job%%;*}"
    fi
    ;;
  valid)
    submit_valid "$@"
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
    samples="${CLI_PROGRESS_SAMPLES:-${PROGRESS_SAMPLES:-1000}}"
    top="${CLI_PROGRESS_TOP:-${PROGRESS_TOP:-1}}"
    [[ "$samples" =~ ^[0-9]+$ ]] || { echo "--samples must be 0 or more, got $samples" >&2; exit 1; }
    [[ "$top" =~ ^[1-9][0-9]*$ ]] || { echo "--top must be 1 or more, got $top" >&2; exit 1; }
    # One node; each GPU scores its own share of the checkpoints.
    gpus="${PROGRESS_GPUS:-4}"
    if [ "$gpus" -gt 1 ]; then default_progress_mem=0; else default_progress_mem=64G; fi
    echo "Scoring the checkpoints in $OUTPUT_DIR on $EVAL_DATA_DIR (${PROGRESS_SPLIT:-test}), $samples molecules, top-$top, $gpus GPUs"
    exec sbatch \
      --job-name="progress_$RUN" \
      --nodes=1 \
      --ntasks-per-node="$gpus" \
      --gpus-per-node="$gpus" \
      --cpus-per-task="$CPUS_PER_GPU" \
      --mem="${PROGRESS_MEM:-$default_progress_mem}" \
      --time="${PROGRESS_TIME:-04:00:00}" \
      --export=ALL \
      "$@" slurm/progress.sbatch
    ;;
esac
