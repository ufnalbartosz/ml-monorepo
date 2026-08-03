"""Shared fixtures for the vision-core tests."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=7)


@pytest.fixture
def write_jpeg():
    """Factory writing a solid-colour JPEG, for the decoding tests."""

    def _write(path, size=(16, 16), colour=(255, 0, 0)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, colour).save(path, format="JPEG")
        return path

    return _write


@pytest.fixture
def fake_split_dataset(rng):
    """A dataset dict in the shape `vision_core.training.train` expects."""

    def _make(num_classes: int = 3, per_split: int = 8, img_size: int = 8):
        def split(prefix: str) -> dict:
            cls = np.arange(per_split, dtype=np.int64) % num_classes
            return {
                f"{prefix}_images": rng.random(
                    (per_split, img_size, img_size, 3), dtype=np.float32
                ),
                f"{prefix}_labels": np.eye(num_classes, dtype=np.float32)[cls],
                f"{prefix}_cls": cls,
            }

        dataset = {}
        for prefix in ("train", "valid", "test"):
            dataset.update(split(prefix))
        return dataset

    return _make
