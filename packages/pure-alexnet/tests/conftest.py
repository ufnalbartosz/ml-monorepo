"""Shared fixtures for the pure-alexnet tests.

Everything here is deliberately tiny: the tests must run on a CPU in seconds,
so they use a 32x32 input and a handful of images rather than the 227x227 /
1360-image data-set the training script uses.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=1234)


@pytest.fixture
def fake_raw_data(rng):
    """Factory for ``(images, one_hot_labels)`` laid out class-by-class.

    Mirrors the layout of the real archive - all of class 0, then all of
    class 1, and so on - without downloading anything.
    """

    def _make(num_classes: int = 4, per_class: int = 10, size: int = 8):
        num_images = num_classes * per_class
        images = rng.random((num_images, size, size, 3), dtype=np.float32)
        class_numbers = np.arange(num_images) // per_class
        labels = np.eye(num_classes, dtype=np.float32)[class_numbers]
        return images, labels

    return _make
