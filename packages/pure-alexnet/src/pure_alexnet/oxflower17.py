"""Loader for the Oxford 17-category flower data-set.

This replaces ``tflearn.datasets.oxflower17``, which was the only reason the
package depended on tflearn.  The archive published by VGG contains a flat
``jpg/`` directory holding ``image_0001.jpg`` .. ``image_1360.jpg``, ordered by
class: the first 80 files are the first category, the next 80 the second, and
so on.  That ordering is the only label information the archive carries.

Everything that touches the network or the filesystem is a separate, injectable
function so the label/decoding logic can be tested without downloading 60 MB.
"""

from __future__ import annotations

import tarfile
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

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


def one_hot_encoded(class_numbers_: Sequence[int] | np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode integer class-numbers into a ``[n, num_classes]`` array."""
    return np.eye(num_classes, dtype=np.float32)[np.asarray(class_numbers_)]


def image_paths(jpg_dir: Path | str) -> list[Path]:
    """Return the JPEG paths in ``jpg_dir``, sorted by filename.

    Sorting matters: the filename order *is* the label order.
    """
    jpg_dir = Path(jpg_dir)
    return sorted(p for p in jpg_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))


def load_image(path: Path | str, image_size: tuple[int, int]) -> np.ndarray:
    """Decode one image to ``float32`` in ``[0, 1]`` with shape ``(*image_size, 3)``."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(image_size[::-1], Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def load_images(paths: Sequence[Path | str], image_size: tuple[int, int]) -> np.ndarray:
    """Decode a sequence of images into a ``[n, height, width, 3]`` array."""
    if not paths:
        return np.zeros((0, *image_size, 3), dtype=np.float32)

    return np.stack([load_image(path, image_size) for path in paths])


def download(url: str = DATA_URL, download_dir: Path | str = "17flowers") -> Path:
    """Download ``url`` into ``download_dir`` unless the file is already there."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_dir / url.rsplit("/", 1)[-1]
    if archive_path.exists():
        print(f"Archive already downloaded: {archive_path}")
        return archive_path

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, archive_path)  # noqa: S310 - fixed https URL
    print(f"Saved to {archive_path}")

    return archive_path


def extract(archive_path: Path | str, dest_dir: Path | str) -> Path:
    """Extract the tarball and return the directory holding the JPEG files."""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, mode="r:gz") as tar:
        # filter='data' refuses absolute paths and symlinks escaping dest_dir.
        tar.extractall(dest_dir, filter="data")

    jpg_dir = dest_dir / "jpg"
    if not jpg_dir.is_dir():
        raise FileNotFoundError(f"No 'jpg' directory inside {archive_path}")

    return jpg_dir


def load_data(
    root: Path | str = "17flowers",
    image_size: tuple[int, int] = (227, 227),
    one_hot: bool = True,
    downloader: Callable[..., Path] = download,
    extractor: Callable[..., Path] = extract,
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

    paths = image_paths(jpg_dir)
    if not paths:
        raise FileNotFoundError(f"No JPEG files found in {jpg_dir}")

    images = load_images(paths, image_size)
    labels = class_numbers(len(paths))

    if one_hot:
        labels = one_hot_encoded(labels, NUM_CLASSES)

    return images, labels
