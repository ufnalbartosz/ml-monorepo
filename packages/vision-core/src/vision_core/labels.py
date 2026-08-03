"""One-hot encoding, shared by both packages' loaders."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def one_hot_encoded(
    class_numbers: Sequence[int] | np.ndarray,
    num_classes: int | None = None,
) -> np.ndarray:
    """Encode integer class-numbers as a ``[len(class_numbers), num_classes]`` matrix.

    For example ``one_hot_encoded([2], num_classes=4)`` is ``[[0, 0, 1, 0]]``.

    :param num_classes: if omitted, taken as ``max(class_numbers) + 1``, which
        is only correct when the highest class actually appears in the input.
    """
    class_numbers = np.asarray(class_numbers)

    if num_classes is None:
        num_classes = int(np.max(class_numbers)) + 1

    return np.eye(num_classes, dtype=np.float32)[class_numbers]
