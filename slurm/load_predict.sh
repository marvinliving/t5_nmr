# Load configs/predict/$PREDICT.env: a trained run's model and a dataset to run
# it on. Sourced from the repository root by submit.sh and, in place of
# load_run.sh, by the evaluate, combine and valid jobs, with the variables
# they read set the same way.

if [ -z "${PREDICT:-}" ]; then
  echo "PREDICT is not set" >&2
  exit 1
fi

PREDICT_CONFIG="configs/predict/$PREDICT.env"
if [ ! -f "$PREDICT_CONFIG" ]; then
  echo "No config $PREDICT_CONFIG; available predictions:" >&2
  ls configs/predict | sed 's/\.env$//' >&2
  exit 1
fi

# The run whose model to use, named by its training config. Read on its own
# first, so the run's config can be loaded under the predict config.
RUN=$(unset RUN; source "$PREDICT_CONFIG"; echo "${RUN:-}")
if [ -z "$RUN" ]; then
  echo "$PREDICT_CONFIG must set RUN, the training config of the model to use" >&2
  exit 1
fi
export RUN
source slurm/load_run.sh

# The dataset comes from the predict config only, so every job of a
# prediction agrees on it. The predict config's settings win over the run's.
unset EVAL_DATA_DIR
set -a
source "$PREDICT_CONFIG"
set +a
if [ -z "${EVAL_DATA_DIR:-}" ]; then
  echo "$PREDICT_CONFIG must set EVAL_DATA_DIR, the folder with src-test.txt and tgt-test.txt" >&2
  exit 1
fi

# The model's name in the predictions' file name: the run for its final
# model, otherwise the folder MODEL_PATH names, for example
# xxl_4x1x4_10ep_checkpoint-424760 or xl_pg4_10ep_gb64.
if [ "$MODEL_PATH" = "$OUTPUT_DIR/final_model" ]; then
  MODEL_LABEL="$RUN"
else
  MODEL_LABEL="$(basename "$(dirname "$MODEL_PATH")")"
  if [ "$(basename "$MODEL_PATH")" != final_model ]; then
    MODEL_LABEL="${MODEL_LABEL}_$(basename "$MODEL_PATH")"
  fi
fi
export MODEL_LABEL

# The chunks' results and predictions, which resubmitting resumes from.
export EVAL_RESULTS_DIR="outputs/predict/$PREDICT"
# The combined predictions go next to the spectra they belong to.
if [ "${EVAL_SPLIT:-test}" = validation ]; then split_file=val; else split_file=test; fi
export PREDICTIONS_FILE="$EVAL_DATA_DIR/prd-${split_file}_$MODEL_LABEL.txt"
