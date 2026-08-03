"""Loader for the Oxford 17-category flower data-set.

This replaces ``tflearn.datasets.oxflower17``, which was the only reason the
package depended on tflearn.  The archive published by VGG contains a flat
``jpg/`` directory holding ``image_0001.jpg`` .. ``image_1360.jpg``, ordered by
class: the first 80 files are the first category, the next 80 the second, and
so on.  That ordering is the only label information the archive carries.

Downloading, unpacking and JPEG decoding are generic and live in
:mod:`vision_core`; what is specific to this data-set - the class names and the
position-implies-label rule - is here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from vision_core.archives import download, extract
from vision_core.images import image_paths, load_images
from vision_core.labels import one_hot_encoded

DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"

NUM_CLASSES = 17
IMAGES_PER_CLASS = 80

#: Category names in the order the archive stores them.
CLASS_NAMES: tuple[str, ...] = (
    "Daffodil",
    "Snowdrop",
    "LilyValley",
    "Bluebell",
    "Crocus",
    "Iris",
    "Tigerlily",
    "Tulip",
    "Fritillary",
    "Sunflower",
    "Daisy",
    "ColtsFoot",
    "Dandelion",
    "Cowslip",
    "Buttercup",
    "Windflower",
    "Pansy",
)


def class_numbers(num_images: int, images_per_class: int = IMAGES_PER_CLASS) -> np.ndarray:
    """Return the class-number for each image, derived from its position.

    The archive is sorted by category, so image ``i`` belongs to class
    ``i // images_per_class``.
    """
    if images_per_class <= 0:
        raise ValueError("images_per_class must be positive")

    return np.arange(num_images, dtype=np.int64) // images_per_class


def download_archive(download_dir: Path | str = "17flowers", url: str = DATA_URL) -> Path:
    """Fetch the VGG tarball into ``download_dir``."""
    return download(url, download_dir)


def extract_archive(archive_path: Path | str, dest_dir: Path | str) -> Path:
    """Unpack the tarball and return the directory holding the JPEG files."""
    extract(archive_path, dest_dir)

    jpg_dir = Path(dest_dir) / "jpg"
    if not jpg_dir.is_dir():
        raise FileNotFoundError(f"No 'jpg' directory inside {archive_path}")

    return jpg_dir


def load_data(
    root: Path | str = "17flowers",
    image_size: tuple[int, int] = (227, 227),
    one_hot: bool = True,
    downloader: Callable[..., Path] = download_archive,
    extractor: Callable[..., Path] = extract_archive,
) -> tuple[np.ndarray, np.ndarray]:
    """Download, extract and decode the data-set.

    Returns ``(images, labels)`` where images are ``float32`` in ``[0, 1]``.
    ``downloader`` and ``extractor`` are injectable so tests can hand over a
    local fixture instead of hitting the network.
    """
    root = Path(root)
    jpg_dir = root / "jpg"

    if not jpg_dir.is_dir():
        archive_path = downloader(download_dir=root)
        jpg_dir = extractor(archive_path, root)

    paths = image_paths(jpg_dir, extensions=(".jpg", ".jpeg"))
    if not paths:
        raise FileNotFoundError(f"No JPEG files found in {jpg_dir}")

    images = load_images(paths, image_size)
    labels = class_numbers(len(paths))

    if one_hot:
        labels = one_hot_encoded(labels, NUM_CLASSES)

    return images, labels
