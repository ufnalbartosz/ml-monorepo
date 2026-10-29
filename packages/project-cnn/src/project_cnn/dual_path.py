"""The two-branch network that used to live inside ``main.py``.

The original was written with Pretty Tensor, a TF1-only library that was
archived in 2018.  ``seq.subdivide(2)`` fed the same tensor into two branches
and concatenated their outputs on the channel axis; that is what
:func:`build_dual_path` does with Keras' functional API.

The other thing ``main.py`` did was build the graph twice - once with
``training=True`` and once with ``training=False`` - under a
``tf.variable_scope(reuse=...)``, so the two copies shared weights.  In Keras
one model does both, because :class:`~project_cnn.tools.PreProcessing` and
``Dropout`` read the ``training`` flag Keras threads through ``__call__``.
"""

from __future__ import annotations

import keras
from keras import layers

from project_cnn.tools import PreProcessing

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (32, 32, 3)
DEFAULT_NUM_CLASSES = 20
DEFAULT_IMG_SIZE_CROPPED = 24


def build_dual_path(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    img_size_cropped: int = DEFAULT_IMG_SIZE_CROPPED,
    dense_units: tuple[int, int] = (256, 128),
    name: str = "dual_path",
) -> keras.Model:
    """Build the pre-processing + two-inception-block + MLP network."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    if img_size_cropped > min(input_shape[0], input_shape[1]):
        raise ValueError(
            f"img_size_cropped={img_size_cropped} exceeds the input size {input_shape[:2]}"
        )

    inputs = keras.Input(shape=input_shape, name="input")

    x = PreProcessing(
        img_size_cropped=img_size_cropped,
        num_channels=input_shape[-1],
        name="pre_process",
    )(inputs)

    # First subdivide: 1x1 (batch-normalized) -> 5x5, and 1x1 -> 3x3.
    branch_a = layers.Conv2D(32, 1, padding="same", use_bias=False, name="inception_1_0_1x1")(x)
    branch_a = layers.BatchNormalization(name="inception_1_0_bn")(branch_a)
    branch_a = layers.Activation("relu", name="inception_1_0_relu")(branch_a)
    branch_a = layers.Conv2D(64, 5, padding="same", activation="relu", name="inception_1_0_5x5")(
        branch_a
    )

    branch_b = layers.Conv2D(64, 1, padding="same", activation="relu", name="inception_1_1_1x1")(x)
    branch_b = layers.Conv2D(128, 3, padding="same", activation="relu", name="inception_1_1_3x3")(
        branch_b
    )

    x = layers.Concatenate(axis=-1, name="inception_1_output")([branch_a, branch_b])

    # Second subdivide: 3x3 -> pool, and 5x5 -> pool.
    branch_a = layers.Conv2D(32, 3, padding="same", activation="relu", name="inception_2_0_3x3")(x)
    branch_a = layers.MaxPooling2D(2, strides=2, padding="same", name="inception_2_0_pool")(
        branch_a
    )

    branch_b = layers.Conv2D(64, 5, padding="same", activation="relu", name="inception_2_1_5x5")(x)
    branch_b = layers.MaxPooling2D(2, strides=2, padding="same", name="inception_2_1_pool")(
        branch_b
    )

    x = layers.Concatenate(axis=-1, name="inception_2_output")([branch_a, branch_b])

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units[0], activation="relu", name="layer_fc1")(x)
    x = layers.Dense(dense_units[1], activation="relu", name="layer_fc2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="target")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
