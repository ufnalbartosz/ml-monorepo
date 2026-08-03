"""Shared fixtures for the project-cnn tests.

The real data-set is 12 000 32x32 images behind a 160 MB download; everything
here is synthetic and small enough that the whole suite runs on a CPU in
seconds.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=4321)


@pytest.fixture
def fake_dataset(rng):
    """Factory for a dataset dict shaped exactly like the cached pickle."""

    def _make(num_classes: int = 4, per_split: int = 12, img_size: int = 32):
        def split(prefix: str) -> dict:
            images = rng.random((per_split, img_size, img_size, 3), dtype=np.float32)
            cls = np.arange(per_split, dtype=np.int64) % num_classes
            labels = np.eye(num_classes, dtype=np.float32)[cls]
            return {
                f"{prefix}_images": images,
                f"{prefix}_labels": labels,
                f"{prefix}_cls": cls,
            }

        dataset = {}
        for prefix in ("train", "valid", "test"):
            dataset.update(split(prefix))
        dataset["class_names"] = [f"class_{i}" for i in range(num_classes)]

        return dataset

    return _make


@pytest.fixture
def cifar_class_names():
    """A stand-in for CIFAR-100's 100 fine-label names."""
    return [f"category_{i:02d}" for i in range(100)]
