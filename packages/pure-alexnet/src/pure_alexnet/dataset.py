"""Train/validation/test splits for the Oxford 17-flowers data-set.

The split itself is a pure function of the one-hot label matrix, so it can be
tested against synthetic labels.  Everything that touches the disk or the
network goes through :class:`DataSet`, whose paths and underlying loader are
constructor arguments - a test can point it at ``tmp_path`` and hand it a
loader that returns a handful of random images.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pure_alexnet import oxflower17
from vision_core.cache import read_pickle, write_pickle

DEFAULT_DATASET_PATH = Path("17flowers/dataset.pickle")

#: The 17 category names, in the order the archive stores them.
LABELS: tuple[str, ...] = oxflower17.CLASS_NAMES

SPLIT_KEYS = (
    "train_images",
    "train_labels",
    "test_images",
    "test_labels",
    "valid_images",
    "valid_labels",
)


@dataclass(frozen=True)
class SplitIndices:
    """Row indices of the three splits, disjoint and covering every row."""

    train: list[int]
    test: list[int]
    valid: list[int]

    def __iter__(self):
        # Kept so `train, test, valid = train_test_valid_split(...)` still works.
        return iter((self.train, self.test, self.valid))


def train_test_valid_split(
    raw_labels: np.ndarray,
    test_fraction: float = 0.2,
    valid_fraction: float = 0.1,
) -> SplitIndices:
    """Split row indices class-by-class into train/test/validation.

    ``raw_labels`` is the one-hot label matrix of shape ``[n, num_classes]``.
    Every class contributes the same number of rows to the test- and
    validation-sets, computed from the *average* class size - which matches the
    original behaviour and is exact for this data-set, where every category has
    the same 80 images.
    """
    if raw_labels.ndim != 2:
        raise ValueError(f"raw_labels must be 2-dimensional, got shape {raw_labels.shape}")
    if not 0.0 <= test_fraction + valid_fraction <= 1.0:
        raise ValueError("test_fraction + valid_fraction must be in [0, 1]")

    images_number, class_number = raw_labels.shape
    per_class = images_number / class_number

    test_budget = dict.fromkeys(range(class_number), int(per_class * test_fraction))
    valid_budget = dict.fromkeys(range(class_number), int(per_class * valid_fraction))

    train_indexes: list[int] = []
    test_indexes: list[int] = []
    valid_indexes: list[int] = []

    for row, label in enumerate(raw_labels):
        index = int(np.argmax(label))

        if test_budget[index] > 0:
            test_budget[index] -= 1
            test_indexes.append(row)
        elif valid_budget[index] > 0:
            valid_budget[index] -= 1
            valid_indexes.append(row)
        else:
            train_indexes.append(row)

    return SplitIndices(train=train_indexes, test=test_indexes, valid=valid_indexes)


def split_dataset(
    raw_images: np.ndarray,
    raw_labels: np.ndarray,
    test_fraction: float = 0.2,
    valid_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Apply :func:`train_test_valid_split` and materialise the six arrays."""
    if len(raw_images) != len(raw_labels):
        raise ValueError(
            f"images and labels disagree on length: {len(raw_images)} vs {len(raw_labels)}"
        )

    splits = train_test_valid_split(raw_labels, test_fraction, valid_fraction)

    return {
        "train_images": raw_images[splits.train],
        "train_labels": raw_labels[splits.train],
        "test_images": raw_images[splits.test],
        "test_labels": raw_labels[splits.test],
        "valid_images": raw_images[splits.valid],
        "valid_labels": raw_labels[splits.valid],
    }


class DataSet:
    """The 17-flowers data-set, cached on disk as a single pickle.

    :param dataset_path: where the prepared pickle lives.  Its parent directory
        also holds the downloaded archive and the extracted ``jpg/`` folder.
    :param loader: callable returning ``(images, one_hot_labels)``.  Defaults to
        downloading from VGG; inject a stub in tests.
    :param image_size: images are resized to this before they are cached.
    """

    def __init__(
        self,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        loader: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
        image_size: tuple[int, int] = (227, 227),
        test_fraction: float = 0.2,
        valid_fraction: float = 0.1,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.loader = loader if loader is not None else oxflower17.load_data
        self.image_size = image_size
        self.test_fraction = test_fraction
        self.valid_fraction = valid_fraction
        self.labels = LABELS

    @property
    def root(self) -> Path:
        """Directory holding the pickle, the archive and the extracted images."""
        return self.dataset_path.parent

    def load(self) -> dict[str, np.ndarray]:
        """Return the six split arrays, building and caching them if needed."""
        if self.dataset_path.exists():
            print("Loading pickle dataset")
            return self.read_pickle()

        print("Creating pickle dataset")
        raw_images, raw_labels = self.loader(root=self.root, image_size=self.image_size)

        dataset = split_dataset(
            raw_images,
            raw_labels,
            test_fraction=self.test_fraction,
            valid_fraction=self.valid_fraction,
        )

        self.write_pickle(dataset)
        self.cleanup()

        return dataset

    def read_pickle(self) -> dict[str, np.ndarray]:
        return read_pickle(self.dataset_path)

    def write_pickle(self, dataset: dict[str, np.ndarray]) -> None:
        write_pickle(dataset, self.dataset_path)

    def cleanup(self) -> None:
        """Delete the raw archive and extracted images once the pickle exists."""
        jpg_dir = self.root / "jpg"
        if jpg_dir.is_dir():
            shutil.rmtree(jpg_dir)

        for name in ("17flowers.tgz", "17flowers.pkl"):
            path = self.root / name
            if path.exists():
                path.unlink()
