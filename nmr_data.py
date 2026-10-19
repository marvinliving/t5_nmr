"""NMR/SMILES data loading and tokenization shared by training and evaluation."""

import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_from_disk


REPO_DIR = Path(__file__).resolve().parent

DEFAULT_PREFIX = "predict SMILES from NMR spectrum: "

# Dataset split name -> file name part, e.g. src-val.txt for "validation".
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}

# How a dataset writes its SMILES: as space-separated tokens, "C C ( = O ) O",
# as alberts_2d and nmr_expt_data do, or without spaces, "CC(=O)O", as in a
# copy made with detokenize_smiles.py. A model writes SMILES the way the
# targets it was trained on do.
SPACINGS = ("spaced", "nospace")


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def split_size(data_dir: Path, split: str) -> int:
    """Number of examples in a split, without loading the spectra."""
    with (data_dir / f"tgt-{SPLIT_FILES[split]}.txt").open("rb") as file:
        return sum(1 for _ in file)


def load_split(data_dir: Path, split: str) -> Dataset:
    """Load one split as a Dataset with "src" (spectrum) and "tgt" (SMILES)."""
    name = SPLIT_FILES[split]
    src_path = data_dir / f"src-{name}.txt"
    tgt_path = data_dir / f"tgt-{name}.txt"
    src_lines = read_lines(src_path)
    tgt_lines = read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            "Line count mismatch:\n"
            f"{src_path}: {len(src_lines)} lines\n"
            f"{tgt_path}: {len(tgt_lines)} lines"
        )
    return Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})


def smiles_spacing(lines: Iterable[str], sample: int = 1000) -> str | None:
    """"spaced" or "nospace", by majority over the first sample SMILES; None if there are none.

    In a spaced file only single-token SMILES such as "C" have no space, and
    a stray space in a no-space file is outvoted.
    """
    spaced = total = 0
    for line in lines:
        if not line.strip():
            continue
        total += 1
        spaced += len(line.split()) > 1
        if total == sample:
            break
    if not total:
        return None
    return "spaced" if 2 * spaced > total else "nospace"


def target_spacing(data_dir: Path, split: str) -> str | None:
    """smiles_spacing of tgt-<split>.txt, reading only its first lines."""
    with (data_dir / f"tgt-{SPLIT_FILES[split]}.txt").open("r", encoding="utf-8") as file:
        return smiles_spacing(file)


def model_spacing(run_dir: Path, setting: str = "auto") -> tuple[str | None, str]:
    """How a run's model writes SMILES, and how that is known.

    setting (SMILES_SPACING) wins unless it is "auto". Otherwise it is the
    spacing of the targets the model was trained on, found through the
    data_dir in the run's run_config.json; None if that can't be read.
    """
    if setting != "auto":
        return setting, "set by SMILES_SPACING"
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return None, f"no run_config.json in {run_dir}"
    data_dir = json.loads(config_path.read_text()).get("data_dir")
    if not data_dir:
        return None, f"no data_dir in {config_path}"
    # A relative DATA_DIR is relative to the repository root.
    data_dir = REPO_DIR / data_dir
    try:
        return target_spacing(data_dir, "train"), f"from the training targets in {data_dir}"
    except FileNotFoundError:
        return None, f"no training targets in {data_dir}"


def spacing_rule(model: str | None, model_source: str, references: str | None) -> tuple[bool, str]:
    """Whether to compare SMILES without spaces, and a line saying so for the log.

    They are compared as written when the model and the references write
    SMILES alike. Otherwise, or if either is unknown, spaces are removed
    from both: they are never part of a SMILES, only of the tokenization.
    """
    ignore_spaces = model is None or model != references
    how = "compared without spaces" if ignore_spaces else "compared as written"
    return ignore_spaces, (
        f"SMILES: model {model or 'unknown'} ({model_source}), "
        f"references {references or 'unknown'}: {how}"
    )


def without_spaces(smiles: str) -> str:
    return "".join(smiles.split())


def build_generation_collate(tokenizer, prefix: str):
    """Collate raw examples into (encoded inputs, reference SMILES) for generate."""

    def collate(examples):
        encoded = tokenizer(
            [prefix + example["src"] for example in examples],
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        targets = [example["tgt"].strip() for example in examples]
        return dict(encoded), targets

    return collate


def tokenize_splits(
    data_dir: Path,
    tokenizer,
    prefix: str,
    target_max_length: int,
    num_proc: int,
) -> DatasetDict:
    """Tokenize the train and validation splits for training.

    The test split is only used for generation, which tokenizes raw text.
    Inputs are never truncated; targets are cut at target_max_length tokens.
    """

    def preprocess(examples):
        model_inputs = tokenizer(
            [prefix + source for source in examples["src"]],
            truncation=False,
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=target_max_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    splits = DatasetDict(
        {split: load_split(data_dir, split) for split in ("train", "validation")}
    )
    return splits.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
        num_proc=max(num_proc, 1),
        desc="Tokenizing dataset",
    )


def load_tokenized_splits(
    data_dir: Path,
    tokenizer,
    model_name: str,
    prefix: str,
    target_max_length: int,
    cache_dir: Path,
    num_proc: int,
    is_main_process: bool = True,
    main_process_first=nullcontext,
) -> DatasetDict:
    """Tokenize once into an Arrow cache on disk, then memory-map it.

    Only the main process tokenizes; under torchrun the others wait inside
    main_process_first() and then load the finished cache. load_from_disk
    memory-maps the Arrow files, so the processes on a node share one copy of
    the data. A later job with the same settings skips tokenization.
    """
    fingerprint = {
        "model_name": model_name,
        "prefix": prefix,
        "target_max_length": target_max_length,
        "data_dir": str(data_dir.resolve()),
        "splits": ["train", "validation"],
    }
    fingerprint_path = cache_dir / "t5_train_fingerprint.json"

    with main_process_first():
        if is_main_process:
            if (
                fingerprint_path.is_file()
                and json.loads(fingerprint_path.read_text()) == fingerprint
            ):
                print("Using tokenized dataset cache:", cache_dir)
            else:
                # Write to a temporary folder and rename it at the end, so an
                # interrupted job never leaves a cache that looks complete.
                partial_dir = cache_dir.with_name(cache_dir.name + ".partial")
                shutil.rmtree(partial_dir, ignore_errors=True)
                tokenize_splits(
                    data_dir, tokenizer, prefix, target_max_length, num_proc
                ).save_to_disk(str(partial_dir))
                (partial_dir / fingerprint_path.name).write_text(
                    json.dumps(fingerprint, indent=4)
                )
                shutil.rmtree(cache_dir, ignore_errors=True)
                partial_dir.rename(cache_dir)
                print("Tokenized dataset saved to:", cache_dir)

        return load_from_disk(str(cache_dir))
