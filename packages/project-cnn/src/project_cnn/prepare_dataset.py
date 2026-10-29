"""Build the train/validation/test splits and cache them as one pickle.

The split itself is a pure function of the test-set arrays, so it can be
checked against synthetic data.  Path handling moved from ``os.getcwd()`` plus
a hard-coded relative name into explicit arguments, which is what lets the
tests run against ``tmp_path``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from project_cnn import loader
from project_cnn.loader import DEFAULT_CONFIG, Cifar100Config

DEFAULT_DATASET_PATH = Path("data/dataset.pickle")

DATASET_KEYS = (
    "train_images",
    "train_labels",
    "train_cls",
    "valid_images",
    "valid_labels",
    "valid_cls",
    "test_images",
    "test_labels",
    "test_cls",
    "class_names",
)


def load_dataset(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    config: Cifar100Config = DEFAULT_CONFIG,
) -> dict:
    """Return the cached splits, downloading and building them if needed."""
    dataset_path = Path(dataset_path)

    if dataset_path.exists():
        return read_pickle(dataset_path)

    return create_dataset(dataset_path, config)


def read_pickle(dataset_path: Path | str) -> dict:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"File '{dataset_path}' does not exist.")

    print("Data has been already downloaded and unpacked.")
    with dataset_path.open("rb") as fp:
        return pickle.load(fp)


def write_pickle(dataset: dict, dataset_path: Path | str) -> Path:
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open("wb") as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_path


def create_dataset(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    config: Cifar100Config = DEFAULT_CONFIG,
) -> dict:
    """Download CIFAR-100, build the three splits and cache them."""
    loader.maybe_download_and_extract(config)

    test_images, test_cls, test_labels = loader.load_data("test", config)
    dataset = split_test_dataset(test_images, test_cls, test_labels)

    train_images, train_cls, train_labels = loader.load_data("train", config)
    dataset["train_images"] = train_images
    dataset["train_labels"] = train_labels
    dataset["train_cls"] = train_cls

    dataset["class_names"] = list(config.class_names)

    write_pickle(dataset, dataset_path)

    return dataset


def split_test_dataset(
    test_images: np.ndarray,
    test_cls: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Halve CIFAR's test-set into a validation-set and a smaller test-set.

    Every other row of each class goes to validation, so both halves keep the
    original class balance.
    """
    halved = np.array([], dtype=int)

    for class_number in range(test_labels.shape[1]):
        halved = np.concatenate((halved, np.where(test_cls == class_number)[0][::2]))
    mask = np.sort(halved)

    # Everything that did not go into the validation-set stays in the test-set.
    # Derive it from the actual size of the test-set instead of a hard-coded
    # 2000, which silently breaks when the list of labels in loader.py changes.
    reversed_mask = np.setdiff1d(np.arange(len(test_cls)), mask)

    return {
        "test_images": test_images[reversed_mask, ...],
        "test_labels": test_labels[reversed_mask, ...],
        "test_cls": test_cls[reversed_mask, ...],
        "valid_images": test_images[mask, ...],
        "valid_labels": test_labels[mask, ...],
        "valid_cls": test_cls[mask, ...],
    }


if __name__ == "__main__":
    load_dataset()
