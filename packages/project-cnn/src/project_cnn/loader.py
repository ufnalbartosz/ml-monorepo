"""Load the 20-class subset of CIFAR-100 this project trains on.

Two things were broken here beyond the TF1 port:

* ``urllib.urlretrieve`` is Python 2; on Python 3 it lives in
  ``urllib.request``.
* the CIFAR-100 pickles were written by Python 2, so on Python 3 every key
  comes back as ``bytes`` and ``data['data']`` raises ``KeyError``.

Both are fixed below.  The data-set location is a :class:`Cifar100Config`
argument rather than a module global, and every transformation - image
reshaping, label remapping, subset masking - is a pure function that can be
tested against a handful of synthetic rows.
"""

from __future__ import annotations

import pickle
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from project_cnn.dataset import one_hot_encoded

DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
DEFAULT_DATA_PATH = Path("data/CIFAR-100/")

# The four CIFAR-100 super-classes this project keeps.
food_containers = ["bottle", "bowl", "can", "cup", "plate"]
fruit_and_vegetables = ["apple", "mushroom", "orange", "pear", "sweet_pepper"]
household_electrical_devices = ["clock", "keyboard", "lamp", "telephone", "television"]
household_furniture = ["bed", "chair", "couch", "table", "wardrobe"]

labels = sorted(
    food_containers + fruit_and_vegetables + household_electrical_devices + household_furniture
)

# Width and height of each image.
img_size = 32

# Number of channels in each image: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = len(labels)


@dataclass(frozen=True)
class Cifar100Config:
    """Where the data lives and which classes to keep."""

    data_path: Path = DEFAULT_DATA_PATH
    data_url: str = DATA_URL
    class_names: tuple[str, ...] = field(default_factory=lambda: tuple(labels))
    img_size: int = img_size
    num_channels: int = num_channels

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def extracted_dir(self) -> Path:
        return Path(self.data_path) / "cifar-100-python"


DEFAULT_CONFIG = Cifar100Config()


def decode_bytes(obj):
    """Recursively turn the ``bytes`` a Python 2 pickle yields into ``str``.

    NumPy arrays pass through untouched, so the image data is not mangled.
    """
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    if isinstance(obj, dict):
        return {decode_bytes(key): decode_bytes(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [decode_bytes(item) for item in obj]
    return obj


def unpickle(filename: str, config: Cifar100Config = DEFAULT_CONFIG) -> dict:
    """Read one of the CIFAR-100 pickle files, normalising Python 2 byte keys."""
    file_path = config.extracted_dir / filename

    print("Loading data: " + str(file_path))

    with file_path.open("rb") as fp:
        # encoding='bytes' keeps the raw image buffers intact; decode_bytes then
        # turns the byte keys and the label-name strings back into str.
        data = pickle.load(fp, encoding="bytes")

    return decode_bytes(data)


def convert_images(
    raw: np.ndarray,
    image_size: int = img_size,
    channels: int = num_channels,
) -> np.ndarray:
    """Reshape flat CIFAR rows to ``[n, height, width, channel]`` floats in [0, 1]."""
    raw_float = np.asarray(raw, dtype=np.float32) / 255.0

    images = raw_float.reshape([-1, channels, image_size, image_size])

    # CIFAR stores channel-first; TensorFlow wants channel-last.
    return images.transpose([0, 2, 3, 1])


def subset_mask(
    fine_labels: np.ndarray,
    class_names: Sequence[str],
    keep: Sequence[str],
) -> np.ndarray:
    """Boolean mask selecting the rows whose class is in ``keep``."""
    wanted = set(keep)
    return np.array([class_names[label] in wanted for label in fine_labels], dtype=bool)


def convert_labels(
    fine_labels: np.ndarray,
    class_names: Sequence[str],
    keep: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the classes outside ``keep`` and renumber the survivors.

    Returns ``(labels, mask)``, where the labels are indices into ``keep`` and
    the mask selects the rows they belong to.
    """
    keep = list(keep)
    mask = subset_mask(fine_labels, class_names, keep)

    converted = np.array(
        [keep.index(class_names[label]) for label in np.asarray(fine_labels)[mask]],
        dtype=np.int64,
    )

    return converted, mask


def load_data(filename: str, config: Cifar100Config = DEFAULT_CONFIG):
    """Load one CIFAR-100 file, subset to the configured classes.

    Returns ``(images, class_numbers, one_hot_labels)``.
    """
    data = unpickle(filename, config)

    class_names = load_class_names(config)
    fine_labels = np.array(data["fine_labels"])

    converted_labels, mask = convert_labels(fine_labels, class_names, config.class_names)

    images = convert_images(data["data"], config.img_size, config.num_channels)[mask, ...]

    return (
        images,
        converted_labels,
        one_hot_encoded(class_numbers=converted_labels, num_classes=config.num_classes),
    )


def load_class_names(config: Cifar100Config = DEFAULT_CONFIG) -> list[str]:
    """The 100 fine-grained CIFAR-100 category names, in label order."""
    # The meta-file holds two lists: 'fine_label_names' (100 classes) and
    # 'coarse_label_names' (20 super-classes). Ask for the one we need instead
    # of relying on the ordering of the dict-keys.
    meta = unpickle("meta", config)
    return meta["fine_label_names"]


def print_download_progress(count: int, block_size: int, total_size: int) -> None:
    pct_complete = float(count * block_size) / total_size

    # The \r means the line overwrites itself.
    sys.stdout.write(f"\r- Download progress: {pct_complete:.1%}")
    sys.stdout.flush()


def download_and_extract(url: str, download_dir: Path | str) -> Path:
    """Download ``url`` into ``download_dir`` and unpack it, unless already there."""
    download_dir = Path(download_dir)
    file_path = download_dir / url.rsplit("/", 1)[-1]

    if file_path.exists():
        print("Data has apparently already been downloaded and unpacked.")
        return file_path

    download_dir.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(  # noqa: S310 - fixed https URL
        url=url,
        filename=file_path,
        reporthook=print_download_progress,
    )

    print()
    print("Download finished. Extracting files.")

    name = str(file_path)
    if name.endswith(".zip"):
        with zipfile.ZipFile(file_path, mode="r") as archive:
            archive.extractall(download_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(file_path, mode="r:gz") as archive:
            archive.extractall(download_dir, filter="data")

    print("Done.")

    return file_path


def maybe_download_and_extract(config: Cifar100Config = DEFAULT_CONFIG) -> Path:
    return download_and_extract(url=config.data_url, download_dir=config.data_path)
