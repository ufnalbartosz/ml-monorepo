"""Decoding image files into numpy arrays."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def image_paths(
    directory: Path | str,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
) -> list[Path]:
    """Image files in ``directory``, sorted by filename.

    Sorting is not cosmetic: data-sets that encode the label in the file
    ordering - the Oxford flowers archive is one - depend on it.
    """
    directory = Path(directory)
    wanted = tuple(ext.lower() for ext in extensions)

    return sorted(p for p in directory.iterdir() if p.suffix.lower() in wanted)


def load_image(path: Path | str, image_size: tuple[int, int]) -> np.ndarray:
    """Decode one image to ``float32`` in ``[0, 1]`` with shape ``(*image_size, 3)``."""
    with Image.open(path) as img:
        # PIL takes (width, height); image_size is (height, width) like the arrays.
        img = img.convert("RGB").resize(image_size[::-1], Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def load_images(paths: Sequence[Path | str], image_size: tuple[int, int]) -> np.ndarray:
    """Decode a sequence of images into a ``[n, height, width, 3]`` array."""
    if not paths:
        return np.zeros((0, *image_size, 3), dtype=np.float32)

    return np.stack([load_image(path, image_size) for path in paths])
