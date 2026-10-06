# NMR-to-SMILES with FLAN-T5

This project fine-tunes Google's [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) models to predict a molecule's structure, written as a SMILES string, from a text encoding of its NMR spectrum. The task is framed as text-to-text translation:

    input:  "predict SMILES from NMR spectrum: <NMR spectrum as text>"
    output: "<SMILES string>"

The best model so far, FLAN-T5-XL trained for 10 epochs, predicts the exact SMILES string for **88.2%** of the 79,441 test molecules.

## Workflow

    ┌────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌───────────────────┐
    │ 1. Set up  │ → │ 2. Train         │ → │ 3. Evaluate         │ → │ 4. Combine        │
    │ env + data │   │ ./submit.sh      │   │ ./submit.sh         │   │ scripts/          │
    │            │   │   train <run>    │   │   evaluate <run>    │   │ combine_results.py│
    └────────────┘   └────────┬─────────┘   └──────────┬──────────┘   └───────────────────┘
                              │                        │
                         t5_train.py        evaluate_exact_match.py
                              │                        │
                              └─→ outputs/<run>/final_model ─┘

A *run* is one training configuration, described by a small file `configs/train/<run>.env`. Every step takes the run name, and all of a run's files go to one output folder. Every step runs as a Slurm job on a GPU cluster.

### 1. Set up the environment and data

Create the Conda environment (Python 3.11, PyTorch 2.12 with CUDA 12.6, Transformers 5.12):

    conda env create -f environment.yml
    conda activate t5

`environment.yml` installs `requirements.txt`, which lists only the packages the code uses. To reproduce the exact package set used on Isambard-AI (linux-aarch64), run `pip install -r environment.lock.txt` in the environment instead.

The dataset isn't stored in Git because of its size and possible redistribution restrictions. On Isambard-AI it is in `/projects/b5an/alberts_2d`, and jobs read it from there: `DATA_DIR` in `slurm/env.sh` sets the location. Elsewhere, point `DATA_DIR` at your copy. The folder needs six files:

    alberts_2d/
      src-train.txt   tgt-train.txt   # 679,213 pairs
      src-val.txt     tgt-val.txt     #  35,749 pairs
      src-test.txt    tgt-test.txt    #  79,441 pairs

Line *N* of each `src-*.txt` file is an NMR spectrum, and line *N* of the matching `tgt-*.txt` file is its SMILES string. Check your copy against the checksums in `data/manifest.tsv`:

    cd /projects/b5an && sha256sum alberts_2d/*.txt

Check the cluster settings in `slurm/env.sh`: the partition, the CUDA module, and `CONDA_ROOT` (default `~/miniforge3`). Then check that a GPU node works before using hours of GPU time:

    ./submit.sh test-gpu

Before the first multi-GPU run, check that all 4 GPUs of a node can communicate (see [Multi-GPU training](#multi-gpu-training)):

    ./submit.sh test-multi-gpu

Check that no reference SMILES is longer than training (`TARGET_MAX_LENGTH`) or evaluation (`GENERATION_MAX_NEW_TOKENS`) allows, in both datasets (`CHECK_DATA_DIRS` in `slurm/env.sh`). The job runs `scripts/check_target_lengths.py` without a GPU, lists any SMILES over a limit, and fails if it finds one; such SMILES can never be predicted exactly:

    ./submit.sh check-lengths

### 2. Train a model

Pick a run from `configs/train/` and submit it:

    ./submit.sh train xl_4x1x4_10ep

`submit.sh` requests the nodes, GPUs, memory and time the config needs, and writes the Slurm log to `logs/`. The job:

1. Downloads the pretrained FLAN-T5 model from Hugging Face. The first run needs internet access. After that, set `LOCAL_FILES_ONLY=1` for offline compute nodes.
2. Tokenizes the training and validation data once and caches it in `outputs/<run>/tokenized/`. Resubmitted jobs reuse the cache.
3. Writes the complete configuration to `outputs/<run>/run_config.json`.
4. Fine-tunes the model, measuring validation loss on the first 5,000 validation molecules after each epoch.
5. Saves a checkpoint every 2,000 steps to `outputs/<run>/checkpoint-*`, keeping the latest two.
6. Saves the finished model to `outputs/<run>/final_model`.
7. Scores exact match on the first 1,000 test molecules as a quick check, saved to `outputs/<run>/test_results.json`. This is not the official score; that comes from step 3.

**Long runs finish on their own.** Each job has a one-day time limit. Twenty minutes before the limit (`STOP_MARGIN_MINUTES`), training saves a checkpoint and stops. The job then submits itself again, and the new job resumes from that checkpoint. This repeats until the run is finished, up to `MAX_RESUBMITS` times (default 20). Submitting a finished run again does nothing, so it never overwrites `final_model`.

A multi-GPU run can only resume on the same number of GPUs, with the same `PARALLEL_MODE`, as the job that wrote the checkpoint. Don't change a run's config while it is in progress.

**Smoke test.** To test a configuration end to end before a long run, set `MAX_STEPS`:

    MAX_STEPS=50 ./submit.sh train xxl_2x2x4_10ep

A smoke test writes to its own folder, `outputs/<run>_max50steps`, so the real run never resumes from its checkpoints.

**Is it still improving?** To decide whether a run needs more training, score its saved models on validation molecules:

    ./submit.sh progress xl_4x1x4_10ep

This one-GPU job scores greedy top-1 exact match on the first 2,000 validation molecules (`PROGRESS_SAMPLES`) for every `checkpoint-*` in the run's folder and for `final_model`. It prints a table with each one's epoch, step, validation loss and exact match, and the change from the one before:

    model           epoch  step   eval_loss  exact_match  samples  change
    checkpoint-<N>  1.0    <N>    <loss>     <score>      2000
    checkpoint-<M>  2.0    <M>    <loss>     <score>      2000     <difference>

If exact match is still rising between the last checkpoints, more epochs would probably help; if it has levelled off, they won't. The validation loss is only shown for checkpoints saved at the end of an epoch, where it is measured. By default it uses validation molecules, so the test set stays unused until the final evaluation.

To follow the score on an external test set instead, such as the experimental spectra in `nmr_expt_data`, name the dataset and split, and how many molecules to score (`0` scores all of them):

    EVAL_DATA_DIR=/projects/b5an/nmr_expt_data PROGRESS_SPLIT=test PROGRESS_SAMPLES=5000 ./submit.sh progress xl_4x1x4_10ep
    EVAL_DATA_DIR=/projects/b5an/nmr_expt_data PROGRESS_SPLIT=test PROGRESS_SAMPLES=0 ./submit.sh progress xl_4x1x4_10ep

Scores are saved to `outputs/<run>/progress/<dataset>_<split>/`, for example `progress/nmr_expt_data_test/`: one JSON file and one predictions file per model, plus `progress.tsv` with the table. Each dataset and split keeps its own scores and table. A score is kept after training deletes its checkpoint (`SAVE_TOTAL_LIMIT` keeps only the latest two), and a later job only scores models it hasn't scored yet. So submitting the job after each epoch, while training goes on, builds the whole curve. It reads checkpoints from `ddp` and `fsdp` runs alike: an FSDP checkpoint, stored as one shard per GPU, is loaded into a single model on one GPU. A checkpoint deleted by training while the job reads it is skipped.

Predictions are written as they are made, so if the job reaches its time limit (`PROGRESS_TIME`, 4 hours), submit it again: it continues the interrupted model where it stopped. Scoring all 61,776 test molecules of `nmr_expt_data` for several checkpoints can take more than one job.

### 3. Evaluate the model on the full test set

    ./submit.sh evaluate xl_4x4_10ep

By default this is one job on one node with 4 GPUs. It splits the test molecules into 4 chunks and evaluates them in parallel, one per GPU. Chunks are rounded up to multiples of 1,000, so the 79,441 test molecules split at 0, 20,000, 40,000 and 60,000. Each chunk writes two files to the run's output folder:

- `test_<start>_<end>_results.json`: top-1 to top-N exact-match scores. While a chunk is running, it saves progress to a matching `.partial.json` file.
- `test_<start>_<end>_predictions.txt`: the model's N most likely SMILES for each spectrum, one per line, best first. Spectrum *i* in the chunk occupies lines *i*·N + 1 to (*i* + 1)·N. This is the same layout as `prd-test.txt` from earlier models.

**GPUs and nodes.** The job is a Slurm array of `EVAL_CHUNKS` tasks (default 1), each on one node with `EVAL_GPUS` GPUs (default 4), and the split is cut into `EVAL_CHUNKS` × `EVAL_GPUS` chunks. Each GPU's log is `logs/eval_<run>_<job>_<task>_gpu<n>.out`. `EVAL_CHUNKS=4` uses 4 nodes and 16 chunks of 5,000; `EVAL_CHUNKS=4 EVAL_GPUS=1` is the old layout, 4 single-GPU tasks, with the same chunks as the default. Resubmit with the same two values to resume, since they set the chunk boundaries.

**Speed.** Spectra are generated in batches of 16 (`EVAL_GENERATION_BATCH_SIZE`). Within each group of 100 batches, spectra are sorted by length so each batch needs little padding. Predictions are still written in dataset order.

**Resuming.** If a chunk runs out of time, submit the evaluation again. Each chunk continues from the spectra already in its predictions file. A finished chunk only rewrites its results.

**Number of outputs per spectrum:** `NUM_OUTPUTS` (default 10).

- `NUM_OUTPUTS=1` uses greedy decoding and only reports top-1.
- `NUM_OUTPUTS=N` (N > 1) uses beam search with N beams and returns the N highest-scoring SMILES. Top-*n* accuracy counts a spectrum as correct if the reference SMILES is anywhere in the first *n* outputs. Beam search can also change the top-1 prediction, so its top-1 score may differ slightly from greedy decoding.
- More outputs make evaluation slower and use more GPU memory. If a chunk runs out of memory, lower `NUM_OUTPUTS` or `EVAL_GENERATION_BATCH_SIZE`.

Set these like any other variable, either in the run's config or at submission, for example `NUM_OUTPUTS=1 ./submit.sh evaluate xl_4x4_10ep`.

A prediction counts as correct only if it is **exactly** the same string as the reference SMILES.

To test a trained model on just 10 molecules first, run `./submit.sh check <run>`.

**External test set.** `EVAL_DATA_DIR` evaluates the final model on another dataset, for example the experimental spectra in `nmr_expt_data` for a model trained on `alberts_2d`. `EVAL_MAX_ROWS` sets how many of its test molecules to use, the first ones; leave it unset (or `0`) for all of them:

    EVAL_DATA_DIR=/projects/b5an/nmr_expt_data ./submit.sh evaluate xl_4x1x4_10ep
    EVAL_DATA_DIR=/projects/b5an/nmr_expt_data EVAL_MAX_ROWS=10000 ./submit.sh evaluate xl_4x1x4_10ep

Results on another dataset go to their own folder, `outputs/<run>/eval_<dataset>/` (here `eval_nmr_expt_data/`), so they never overwrite the results on the run's own test set. The combine job records them in the summary as `<run>_on_<dataset>`. `./submit.sh check` reads `EVAL_DATA_DIR` too. Give the same `EVAL_DATA_DIR` and `EVAL_MAX_ROWS` when you submit again to resume.

### 4. Combine the results

`./submit.sh evaluate` also submits a short CPU job, `combine_<run>`, that starts once every chunk has succeeded and runs `scripts/combine_results.py`. Its log is `logs/combine_<run>_<job>.out`. If a chunk fails, Slurm cancels the combine job; submit the evaluation again to resume the chunks and queue a new one. To combine by hand, for example after copying results from another machine:

    python scripts/combine_results.py xl_4x4_10ep

This script:

- checks that the chunks cover the whole test set exactly once;
- prints top-1 to top-N exact match, weighting each chunk by its number of molecules;
- joins the chunks' predictions into `prd-test.txt` in the output folder;
- records top-1 (`exact_match`) to top-N (`top_2`, `top_3`, …) in `reports/evaluation_summary.tsv` under the run name. Combining a run again replaces its row. The combine job names the row after the output folder when `--nodes`, `--gpus` or `MAX_STEPS` gave the run its own folder, for example `xl_pg4_10ep_gb128`; `--experiment` sets the name by hand.

Each chunk's results record the size of the test set. Results from evaluations made before this was added don't, so for those the script counts the test set in the run's `DATA_DIR`, or in the folder given by `--data-dir`. `--total 79441` skips the count.

### 5. Score valid SMILES only

    ./submit.sh valid xl_4x4_10ep
    EVAL_DATA_DIR=/projects/b5an/nmr_expt_data ./submit.sh valid xl_4x4_10ep

Once the combine job has written `prd-test.txt`, this CPU job runs `scripts/valid_smiles.py` on it. It checks every candidate with RDKit and scores top-1 to top-N three ways:

- `raw_exact`: exact match as the evaluation scores it. It should equal the run's row in `reports/evaluation_summary.tsv`.
- `valid_exact`: exact match after dropping candidates RDKit can't parse, so the *n*-th valid candidate moves up to rank *n*.
- `valid_canonical`: as `valid_exact`, but comparing RDKit canonical SMILES, so the same molecule written differently counts, and a molecule repeated among the candidates is kept only once.

It also reports the share of valid candidates, overall and at each rank, and how many reference SMILES RDKit can't parse; those can never score a canonical match. Accuracies are over all spectra. The table is in `logs/valid_<run>_<job>.out` and in `prd-test_valid.json` next to the predictions. A whole node bills the same as one core, so the job takes the node and runs one RDKit worker per core. It needs RDKit in the environment (`pip install rdkit`). To run it by hand, on any predictions file:

    python scripts/valid_smiles.py outputs/<run>/prd-test.txt --data-dir /projects/b5an/alberts_2d

## Runs

Each file in `configs/train/` is one run. It holds only the settings that differ from the defaults, for example `configs/train/xxl_2x2x4_10ep.env`:

    MODEL_NAME=google/flan-t5-xxl
    TRAIN_BATCH_SIZE=2
    GRAD_ACCUMULATION_STEPS=2
    NUM_EPOCHS=10
    GRADIENT_CHECKPOINTING=1
    PARALLEL_MODE=fsdp
    GPUS_PER_NODE=4

Run names are `<model>_<batch>x<accumulation>[x<gpus>]_<epochs>ep`. For example, `xxl_2x2x4_10ep` trains FLAN-T5-XXL with 2 examples per GPU per step, 2 steps of gradient accumulation and 4 GPUs (2 × 2 × 4 = 16 examples per update) for 10 epochs. Single-GPU runs leave out the GPU count: `xl_4x4_10ep` means 4 × 4 on one GPU. Multi-node runs, whose global batch depends on the number of nodes, are named by their per-GPU batch instead: `xl_pg4_10ep` has 4 examples per GPU (see [Multi-node training](#multi-node-training)).

| Model | Parameters | 1 GPU | 4 GPUs (1 node) | Several nodes |
|---|---|---|---|---|
| small | 80M | `small_4x4_3ep` | `small_4x1x4_3ep` | |
| base | 250M | `base_16x1_3ep`, `base_scaling_gb16` | | `base_scaling_gb128_sqrt`, `base_scaling_gb128_linear` |
| large | 780M | `large_4x4_3ep`, `large_4x4_10ep` | `large_4x1x4_10ep` | |
| xl | 3B | `xl_4x4_3ep`, `xl_1x16_3ep`, `xl_4x4_10ep`, `xl_1x16_10ep`, `xl_2x8_10ep` | `xl_4x1x4_10ep` | `xl_pg4_10ep` |
| xxl | 11B | (can't fit, see [Multi-GPU training](#multi-gpu-training)) | `xxl_4x1x4_10ep`, `xxl_2x2x4_10ep`, `xxl_1x4x4_10ep` | `xxl_2x1x8_10ep`, `xxl_pg2_10ep` |

The runs in the first two GPU columns update the model with 16 examples at a time: batch per GPU × accumulation steps × number of GPUs = 16. This keeps them comparable with the 88.2% XL result. Runs with a smaller batch per step, such as 1×16, use less GPU memory but train more slowly. The multi-node runs use larger global batches with a scaled learning rate, which is a different recipe; see [Multi-node training](#multi-node-training).

To add a run, copy a config under a new name, or start from `configs/template.env`, which lists every setting at its default. The run's output folder is `outputs/<run>`.

`probe_xl_4x4_nogc` is not a real run. It trains XL for 200 steps without gradient checkpointing and prints the peak GPU memory, to find out whether single-GPU XL runs need gradient checkpointing at all.

### Runs from before `configs/`

Runs trained with the old per-run scripts keep their output folders; their configs set `OUTPUT_DIR` to the old folder, so they can still be evaluated and resumed:

| Old script | Run | Output folder |
|---|---|---|
| `start_training_small_4x4_3ep.sh` | `small_4x4_3ep` | `outputs/flan-t5-small_nmr_input1536_4x4_3ep` |
| `start_training_base_16x1_3ep.sh` | `base_16x1_3ep` | `outputs/flan-t5-base_nmr_input1536_16x1_3ep` |
| `start_training_large_4x4_3ep.sh` | `large_4x4_3ep` | `outputs/flan-t5-large_nmr_input1536_bs16` |
| `start_training_large_4x4_10ep.sh` | `large_4x4_10ep` | `outputs/flan-t5-large_nmr_input1536_bs4x4_10ep` |
| `start_training_xl_4x4_3ep.sh` | `xl_4x4_3ep` | `outputs/flan-t5-xl_nmr_input1536_bs4x4_3ep` |
| `start_training_xl_1x16_3ep.sh` | `xl_1x16_3ep` | `outputs/flan-t5-xl_nmr_input1536_bs16_gc` |
| `start_training_xl_4x4_10ep.sh` | `xl_4x4_10ep` | `outputs/flan-t5-xl_nmr_input1536_4x4_ep10` |
| `start_training_xl_1x16_10ep.sh` | `xl_1x16_10ep` | `outputs/flan-t5-xl_nmr_input1536_1x16_ep10` |
| `start_training_xl_2x8_10ep.sh` | `xl_2x8_10ep` | `outputs/flan-t5-xl_nmr_input1536_2x8_ep10` |

The single-GPU XXL scripts (`xxl_4x4_10ep`, `xxl_2x8_10ep`, `xxl_1x16_10ep`) were removed because XXL can't train on one GPU. The `evaluate_*_array.sh` and `evaluate_check.sh` scripts are replaced by `./submit.sh evaluate <run>` and `./submit.sh check <run>`.

### Variables

A config sets any of these. Variables that a config doesn't set can also be given at submission, as in `MAX_STEPS=50 ./submit.sh train <run>`; when both set a variable, the config wins. The exceptions are `OUTPUT_DIR`, which comes only from the config or the run name, and `NODES`/`GPUS_PER_NODE`, which `--nodes` and `--gpus` override.

**Training** (read by `t5_train.py` through `config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `google/flan-t5-base` | Hugging Face model to fine-tune |
| `DATA_DIR` | `/projects/b5an/alberts_2d` (from `slurm/env.sh`) | Folder with the `src-*`/`tgt-*` files. A relative path is relative to the repository |
| `OUTPUT_DIR` | `outputs/<run>` | Where checkpoints and the final model go |
| `PER_GPU_BATCH` | 16 | Examples per GPU per step. Older configs call it `TRAIN_BATCH_SIZE`, which still works |
| `GRAD_ACCUMULATION_STEPS` | 1 | Steps combined into each model update |
| `NUM_EPOCHS` | 3 | Passes over the training data |
| `MAX_STEPS` | 0 | Stop after this many model updates, for smoke tests. 0 trains for `NUM_EPOCHS` |
| `LEARNING_RATE` | 5e-5 | Learning rate for a global batch of `BASE_BATCH` |
| `BASE_BATCH` | 16 | The global batch `LEARNING_RATE` belongs to |
| `LR_SCALING` | `none` | How the learning rate follows a larger global batch: `none`, `sqrt` or `linear` (see [Multi-node training](#multi-node-training)) |
| `WARMUP_RATIO` | 0 | Share of training spent raising the learning rate from 0, for example 0.03 |
| `WEIGHT_DECAY` | 0.01 | Optimizer weight decay |
| `GRADIENT_CHECKPOINTING` | 0 | 1 trades speed for lower GPU memory |
| `TARGET_MAX_LENGTH` | 640 | SMILES strings longer than this many tokens are cut off |
| `EVAL_BATCH_SIZE` | 4 × `PER_GPU_BATCH` | Batch size for validation loss |
| `EVAL_MAX_SAMPLES` | 5000 | Validation molecules (the first ones) used for validation loss; 0 uses all 35,749 |
| `GROUP_BY_LENGTH` | 0 | 1 batches spectra of similar length together (see [Performance](#performance-and-comparability)) |
| `DATALOADER_NUM_WORKERS` | 4 on 1 GPU, 8 on several | Worker processes per GPU that prepare batches; also used to tokenize the data |
| `SAVE_STEPS` / `SAVE_TOTAL_LIMIT` | 2000 / 2 | Checkpoint frequency and how many to keep |
| `SAVE_STRATEGY` | steps | `epoch` saves at the end of every epoch instead of every `SAVE_STEPS` updates |
| `STOP_MARGIN_MINUTES` | 20 | Save and stop this long before the job's time limit |
| `TEST_SAMPLE_SIZE` | 1000 | Test molecules in the quick end-of-training check |
| `GENERATION_BATCH_SIZE` | 32 | Batch size for the quick check |
| `GENERATION_MAX_NEW_TOKENS` | 640 | Longest SMILES the quick check and the evaluation can generate |
| `SEED` | 42 | Random seed |
| `LOCAL_FILES_ONLY` | 0 | Set to 1 on offline nodes after the model is downloaded once |
| `PARALLEL_MODE` | `none`, or `ddp` on several GPUs | `none`, `ddp`, `fsdp` or `hsdp`; see [Which mode to use](#which-mode-to-use) |
| `TOKENIZED_CACHE_DIR` | `<OUTPUT_DIR>/tokenized` | Where the tokenized dataset is cached. It is rebuilt when the model, prefix, `TARGET_MAX_LENGTH` or `DATA_DIR` changes |

NMR inputs are never truncated. T5 has no fixed maximum input length, so the whole spectrum is always used. Very long inputs use a lot of GPU memory; if a job runs out of memory, use a configuration with a smaller batch per step.

**Slurm resources** (read by `submit.sh` and the jobs in `slurm/`):

| Variable | Default | Meaning |
|---|---|---|
| `NODES` / `GPUS_PER_NODE` | 1 / 1 | GPUs to train on; `--nodes` and `--gpus` on `submit.sh` override them. More than one GPU starts one training process per GPU |
| `TRAIN_TIME` | `1-00:00:00` | Time limit of each training job |
| `TRAIN_MEM` | `64G` on 1 GPU, whole node otherwise | Host memory of a training job |
| `MAX_RESUBMITS` | 20 | How many times an unfinished run submits itself again |
| `EVAL_CHUNKS` | 1 | Evaluation array tasks, one node each |
| `EVAL_GPUS` | 4 | GPUs per evaluation array task, each evaluating its own chunk; the split is cut into `EVAL_CHUNKS` × `EVAL_GPUS` chunks. Above 1, a task takes the node's memory (`EVAL_MEM` 0) |
| `EVAL_DATA_DIR` | the run's `DATA_DIR` | Dataset that `evaluate`, `check` and `progress` use, for example an external test set. Results on another dataset go to `outputs/<run>/eval_<dataset>/` (`progress/<dataset>_<split>/` for `progress`) |
| `EVAL_MAX_ROWS` | 0 | Evaluate only the first rows of the split, for example 20000; 0 evaluates all of it |
| `EVAL_TIME` / `EVAL_MEM` | `10:00:00` / `64G` | Time limit and host memory of each evaluation chunk |
| `EVAL_GENERATION_BATCH_SIZE` | 16 | Spectra per `generate` call in the evaluation |
| `NUM_OUTPUTS` | 10 | Candidate SMILES per spectrum in the evaluation |
| `EVAL_SPLIT` | `test` | `test` or `validation` |
| `PROGRESS_SPLIT` | `validation` | Split that `./submit.sh progress` scores: `validation` or `test` |
| `PROGRESS_SAMPLES` | 2000 | Molecules (the first ones of the split) that `./submit.sh progress` scores each model on; 0 uses all |
| `PROGRESS_TIME` | `04:00:00` | Time limit of the progress job. Submitted again, it carries on where it stopped |
| `VALID_TIME` | `02:00:00` | Time limit of `./submit.sh valid` |

Other `sbatch` options can be added after the run name, for example `./submit.sh train xl_4x1x4_10ep --qos=long`. They apply only to the first job, not to the jobs it resubmits.

## Multi-GPU training

An Isambard-AI node has 4 GH200 superchips. Each is one 96 GB GPU with its own 72-core Grace CPU and 120 GB of memory, and the 4 GPUs of a node are connected by NVLink. A run on several GPUs starts **one Slurm task per GPU**, as the Isambard-AI documentation recommends: `--ntasks-per-node` and `--gpus-per-node` equal to the GPUs per node, and `--cpus-per-task=72`. Slurm pins each task to the Grace CPU attached to its GPU, and each task runs one copy of `t5_train.py`. The Trainer connects the copies with NCCL from the `RANK`, `LOCAL_RANK` and `WORLD_SIZE` variables that `slurm/train.sbatch` sets.

**Why XXL needs several GPUs.** Full fine-tuning with the AdamW optimizer in mixed precision keeps about 16 bytes per parameter on the GPU: fp32 weights (4), fp32 gradients (4) and two optimizer moments (8). This is before counting activations, the intermediate results kept for the backward pass.

| Model | Training state | 1 GPU | Split over 4 GPUs | Split over 8 GPUs |
|---|---|---|---|---|
| large (0.78B) | ~12 GB | fits | – | – |
| xl (2.85B) | ~46 GB | fits | ~12 GB | ~6 GB |
| xxl (11.3B) | ~180 GB | never fits | ~45 GB | ~23 GB |

No batch size or gradient checkpointing setting makes XXL fit on one 96 GB GPU.

### Which mode to use

| Model | Mode | What it does |
|---|---|---|
| small, base, large, xl | `ddp` | Every GPU holds a full copy of the model and processes different examples. The GPUs average their gradients after each step: the only communication, so this also works well between nodes. |
| xxl, 1 node | `fsdp` | The weights, gradients and optimizer state are split over the 4 GPUs. Each GPU briefly gathers one T5 block's weights when it needs them, over NVLink. |
| xxl, 2 or more nodes | `hsdp` | Split over the 4 GPUs of each node, as `fsdp`, and copied across nodes, as `ddp`. Only a gradient average crosses the network, instead of gathering 45 GB of weights over it every step. |

All three are the same optimization as a single-GPU run for the same global batch. XL fits on one GPU (46 GB of training state), so it uses `ddp`; without gradient checkpointing it needs more memory for activations, which `probe_xl_4x4_nogc` measures. XXL uses gradient checkpointing to leave room for activations.

`hsdp` needs every task to see all 4 GPUs of its node. `./submit.sh test-multi-gpu` reports what each task sees; if it prints `sees 1 GPU(s)`, use `fsdp` on one node instead.

If `xxl_2x2x4_10ep` runs out of GPU memory, use `xxl_1x4x4_10ep`.

**Checkpoints.** A job can only resume a checkpoint written with the same number of GPUs, `PARALLEL_MODE`, global batch and learning rate. `run_config.json` records them, and a job that doesn't match stops with a message saying what differs. An FSDP or HSDP checkpoint is stored as one shard per GPU. The final model is gathered into a normal Hugging Face model in `final_model/`, which loads on any number of GPUs. For XXL, each checkpoint takes about 135 GB, because it includes the optimizer state. With `SAVE_TOTAL_LIMIT=2` and the 45 GB final model, one XXL run needs about 315 GB of project storage.

**Host memory.** Multi-GPU jobs request the whole node's memory (`--mem=0`). Rank 0 loads the fp32 XXL weights (45 GB) and gathers the full model when saving, and the other processes wait. The tokenized dataset is memory-mapped, so all processes on a node share one copy. If your partition doesn't allow `--mem=0`, set `TRAIN_MEM=360G`.

**Between nodes.** Within a node, GPUs communicate over NVLink with no extra setup. Between nodes, NCCL needs the aws-ofi-nccl plugin to use the Slingshot network. When `NODES` is more than 1, the job calls `setup_multinode_nccl` from `slurm/env.sh`. This loads `brics/nccl` and `brics/aws-ofi-nccl` and sets the variables from the [Isambard-AI NCCL guide](https://docs.isambard.ac.uk/user-documentation/guides/nccl/). Run `./submit.sh test-multi-gpu --nodes=2` first. The log should show `NET/AWS Libfabric`. If it shows `NET/Socket`, NCCL has fallen back to TCP, which works but is slow: fix that before using more than 2 nodes.

## Multi-node training

More GPUs only shorten a run if each step does more work. With the global batch fixed at 16, a 10-epoch XL run is always about 424,000 optimizer steps, and 16 examples can't usefully be split over more than 4–8 GPUs. So multi-node runs grow the global batch with the GPUs, `PER_GPU_BATCH` × GPUs, and take proportionally fewer steps:

| Nodes × GPUs | Per-GPU batch | Global batch | Optimizer steps (10 XL epochs) |
|---|---|---|---|
| 1 × 1 (as before) | 4 (× 4 accumulation) | 16 | 424k |
| 1 × 4 | 4 | 16 | 424k |
| 4 × 4 | 4 | 64 | 106k |
| 16 × 4 | 4 | 256 | 26k |

A larger batch needs a larger learning rate to make the same progress per example. `LEARNING_RATE` stays the value for a global batch of 16 (`BASE_BATCH`), and `LR_SCALING` scales it to the actual global batch:

- `sqrt`: × √(global / 16). At a global batch of 256 this gives 2e-4.
- `linear`: × global / 16. At 256 this gives 8e-4.

`WARMUP_RATIO=0.03` raises the learning rate from 0 over the first 3% of training, which large learning rates usually need. `run_config.json` records both the base and the effective learning rate. This is a different recipe from the earlier runs, so it is validated once before it's used for XL or XXL.

A multi-node config fixes the per-GPU batch, and the number of nodes comes from the command line:

    ./submit.sh train xl_pg4_10ep --nodes 4

This writes to `outputs/xl_pg4_10ep_gb64`: the folder name records the global batch, so different node counts never share checkpoints. `evaluate` and `check` need the same `--nodes` to find it.

**Step 1: validate the recipe** on FLAN-T5-base, which takes a few GPU-hours:

    ./submit.sh train base_scaling_gb16             # reference: global batch 16, 1 GPU
    ./submit.sh train base_scaling_gb128_sqrt       # global batch 128 on 2 nodes, sqrt scaling
    ./submit.sh train base_scaling_gb128_linear     # the same with linear scaling

Then compare them on the first 20,000 test molecules:

    EVAL_MAX_ROWS=20000 ./submit.sh evaluate base_scaling_gb16

The combine job passes `--total 20000` itself.

Repeat for the other two runs. Accept a scaled recipe if its top-1 exact match is within about 0.5% of the reference. If neither is, try a global batch of 64. Then set `LR_SCALING` in `xl_pg4_10ep` and `xxl_pg2_10ep` to the recipe that passed.

**Step 2: find how many nodes are worth it.** Run 100 steps on 1, 2, 4 and 8 nodes:

    for n in 1 2 4 8; do MAX_STEPS=100 ./submit.sh train xl_pg4_10ep --nodes $n; done
    python scripts/scaling_table.py 'outputs/xl_pg4_10ep_gb*_max100steps' --project-examples 6792130

The table shows samples per second, the efficiency per GPU compared with one node, and the hours for 10 epochs. Use the largest node count whose efficiency stays above about 80%, and record the table here.

**Step 3: XXL.** `MAX_STEPS=50 ./submit.sh train xxl_pg2_10ep` runs 50 steps on 2 nodes with `hsdp` and prints the peak GPU memory. Cancel it after a checkpoint and submit again to check it resumes. Then submit the real run with the node count from step 2.

### Checking a new setup

In order:

1. `./submit.sh test-multi-gpu`: all 4 ranks print their GPU, which GPUs they see and their CPUs, and rank 0 prints `all_reduce OK` and the bandwidth.
2. `./submit.sh test-multi-gpu --nodes=2`: the same over 8 ranks, and `NET/AWS Libfabric` in the log.
3. `MAX_STEPS=200 ./submit.sh train small_4x1x4_3ep` and `MAX_STEPS=200 ./submit.sh train small_4x4_3ep`, then compare `train_loss` in their `train_results.json`. They should be close, but not identical, because the examples are processed in a different order.
4. `MAX_STEPS=100 SAVE_STEPS=50 ./submit.sh train xl_4x1x4_10ep`: cancel it once `checkpoint-50` exists and submit it again, to check a multi-GPU job can save and resume. `scancel --signal=USR1 <jobid>` instead of a plain cancel makes it save and stop cleanly, as it does before the time limit.
5. `MAX_STEPS=50 ./submit.sh train xxl_2x2x4_10ep`: at the end, rank 0 prints `Peak GPU memory allocated (GB)`.

## Performance and comparability

The code went through an audit of training and evaluation cost. Changes that leave the optimization untouched (same examples per update, optimizer, learning-rate schedule, precision and loss) are on by default:

- Validation loss uses the first 5,000 validation molecules with a larger batch, instead of all 35,749 at the training batch size. It is only a diagnostic, so `eval_loss` in logs of new jobs isn't directly comparable with older logs.
- Batches are padded to a multiple of 8 tokens for the GPU's tensor cores. Padding is masked, so the loss is unchanged.
- Evaluation sorts spectra by length and uses larger batches. This changes only how spectra are grouped into batches, which can change a prediction in rare cases through rounding. `--no-sort-by-length` on `evaluate_exact_match.py` turns sorting off.
- Without bf16 support, training and evaluation now use fp32 instead of fp16, in which T5 overflows. GH200 GPUs support bf16, so Isambard-AI runs are unaffected.

These changes need a comparison before they become defaults, because they change what the model is trained on:

- `GROUP_BY_LENGTH=1` draws training examples in a different order, grouping similar lengths into each batch. It should save a large share of the compute spent on padding. Validate it on the small model first: compare exact match with and without it.
- Disabling gradient checkpointing for single-GPU XL runs, if `probe_xl_4x4_nogc` shows it fits.
- Longer `TARGET_MAX_LENGTH`, or adding SMILES characters the T5 tokenizer can't represent. `scripts/dataset_stats.py` measures both (see below). Runs with a changed tokenizer aren't comparable with the 88.2% result.

**Dataset statistics.** `python scripts/dataset_stats.py --model-name google/flan-t5-base --data-dir /projects/b5an/alberts_2d` writes `reports/dataset_stats.json`. It needs only the tokenizer, not a GPU. It reports:

- token-length percentiles of spectra and SMILES;
- the share of SMILES longer than `TARGET_MAX_LENGTH`;
- SMILES that the tokenizer can't reproduce (`decode(encode(s)) != s`), and the characters responsible;
- how much padding random and length-grouped batches would need.

Any SMILES in the last two categories can never be predicted exactly.

## Results

Exact match on all 79,441 test molecules, from `reports/evaluation_summary.tsv`. These runs used earlier versions of the scripts, which truncated inputs to the length shown. Current scripts don't truncate inputs, so new runs may score differently.

| Model | 512 tokens | 1024 tokens | 1536 tokens |
|---|---|---|---|
| small, 3 epochs | 15.9% | 29.9% | 36.7% |
| base, 3 epochs | – | 65.1% | 68.9% |
| large, 3 epochs | – | 78.6% | 80.6% |
| xl, 3 epochs | – | 83.8% | 84.9–85.2% |
| **xl, 10 epochs** | – | – | **88.2%** |

Accuracy improves with model size, with longer inputs and with longer training. The large (10 epochs) and XXL runs have not been evaluated.

Rows added by `scripts/combine_results.py` are named after the run. Older rows are named after the Slurm evaluation job. `reports/evaluation_chunks.tsv` has the score for each chunk of the older evaluations. `eval_check_5928345` in the summary is a 10-molecule smoke test and isn't comparable with the full evaluations.

## Tests

    pytest tests/

- `tests/test_units.py` checks the evaluation bookkeeping (resume, chunk tiling, summary rows) and the training setup in about a minute. One test trains a tiny model under FSDP on two CPU processes, saves a sharded checkpoint, and checks that `checkpoint_progress.py` loads exactly those weights.
- `tests/test_smoke.py` trains `google/flan-t5-small` for 3 steps on CPU on a tiny synthetic dataset. It then evaluates, resumes an interrupted evaluation, combines the chunks, and scores the checkpoints with `checkpoint_progress.py`, on its own validation set and on all of a second, external dataset, including resuming an interrupted scoring. It downloads the model on first use and takes a few minutes.

GitHub Actions runs both on every push (`.github/workflows/smoke.yml`). `python tests/make_tiny_dataset.py` writes the tiny dataset to `tmp/tiny/` for trying things by hand.

## Repository layout

    t5_train.py                  training pipeline
    config.py                    training configuration from environment variables
    nmr_data.py                  data loading and tokenization
    evaluate_exact_match.py      evaluation program (also used for the quick test)
    submit.sh                    submits training, evaluation, check and progress jobs
    configs/train/*.env          one file per run
    configs/template.env         every setting at its default
    slurm/env.sh                 cluster-specific settings
    slurm/*.sbatch               Slurm job templates
    scripts/combine_results.py   combines evaluation chunks
    scripts/checkpoint_progress.py  exact match of every checkpoint, on any dataset
    scripts/dataset_stats.py     token-length and tokenizer statistics
    scripts/check_target_lengths.py  SMILES longer than the token limits
    scripts/scaling_table.py     throughput and scaling efficiency across node counts
    tests/                       unit and smoke tests
    requirements.txt             packages the code uses
    environment.yml              Conda environment
    environment.lock.txt         exact package list on Isambard-AI
    data/                        dataset manifest and checksums
    reports/                     evaluation results

These folders are created locally and aren't stored in Git: `outputs/` (models and checkpoints) and `logs/` (Slurm logs). The datasets live outside the repository. `nmr_expt_data` is a second dataset listed in `data/manifest.tsv`, in `/projects/b5an/nmr_expt_data`; to train on it, set `DATA_DIR` to that folder in a run's config, and to test a model trained on another dataset on it, give `EVAL_DATA_DIR` (see [External test set](#3-evaluate-the-model-on-the-full-test-set)).

## Running on another cluster

Everything cluster-specific is in `slurm/env.sh`:

- `SBATCH_PARTITION`: a GPU partition on your cluster. Add `SBATCH_ACCOUNT` if your cluster needs one.
- `setup_job_env`: your cluster's CUDA module, and `CONDA_ROOT` for your Conda installation.
- `CPUS_PER_GPU`: CPU cores per GPU on your nodes.
- `setup_multinode_nccl`: the `brics/*` modules and `NCCL_*`/`FI_*` variables for Isambard-AI's Slingshot network. On InfiniBand clusters NCCL usually needs no plugin; on other networks check your cluster's NCCL documentation.

In the configs, set `GPUS_PER_NODE` to the GPUs per node and keep batch × accumulation × GPUs at 16. If your cluster doesn't allow whole-node memory requests, set `TRAIN_MEM`; see the host memory note under [Multi-GPU training](#multi-gpu-training).
