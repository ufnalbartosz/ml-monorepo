"""Inception-style CNN for the CIFAR-100 subset, ported from tflearn to Keras 3.

The tflearn version built the graph, loaded the data-set and launched a
1000-epoch run at import time.  This module only builds the model; training
lives in :mod:`project_cnn.main`.

The architecture is unchanged: two inception blocks, each with a 1x1 branch, a
1x1 -> 3x3 branch, a 1x1 -> 5x5 branch and a pool -> 1x1 branch, concatenated on
the channel axis.  Layer names carry over from the tflearn graph.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers

from project_cnn.layers import LocalResponseNormalization

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (32, 32, 3)
DEFAULT_NUM_CLASSES = 20


def build_augmentation(max_rotation_degrees: float = 25.0) -> keras.Sequential:
    """Random flip + rotation, the Keras equivalent of tflearn's ImageAugmentation.

    Augmentation layers are inert at inference time, which is what the tflearn
    ``ImageAugmentation`` object arranged by hand.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="random_flip_leftright"),
            layers.RandomRotation(max_rotation_degrees / 360.0, name="random_rotation"),
        ],
        name="augmentation",
    )


def build_normalization(train_images: np.ndarray | None = None) -> layers.Normalization:
    """Feature-wise zero-centre and unit-variance, as tflearn's ImagePreprocessing did.

    Pass the training images to adapt the statistics; leave them out to get an
    un-adapted layer (useful in tests, and adaptable later).
    """
    normalization = layers.Normalization(name="normalization")

    if train_images is not None:
        normalization.adapt(train_images)

    return normalization


def inception_block(x, name: str, reduce_filters: int = 64, branch_filters: int = 32):
    """One inception block: 1x1 / 1x1->3x3 / 1x1->5x5 / pool->1x1, concatenated."""
    branch_1_1 = layers.Conv2D(
        reduce_filters, 1, padding="same", activation="relu", name=f"{name}a_1_1"
    )(x)

    branch_3_3 = layers.Conv2D(
        reduce_filters, 1, padding="same", activation="relu", name=f"{name}b_3_3_reduce"
    )(x)
    branch_3_3 = layers.Conv2D(
        branch_filters, 3, padding="same", activation="relu", name=f"{name}b_3_3"
    )(branch_3_3)

    branch_5_5 = layers.Conv2D(
        reduce_filters, 1, padding="same", activation="relu", name=f"{name}c_5_5_reduce"
    )(x)
    branch_5_5 = layers.Conv2D(
        branch_filters, 5, padding="same", activation="relu", name=f"{name}c_5_5"
    )(branch_5_5)

    branch_pool = layers.MaxPooling2D(3, strides=1, padding="same", name=f"{name}d_pool")(x)
    branch_pool = layers.Conv2D(
        branch_filters, 1, padding="same", activation="relu", name=f"{name}d_pool_1_1"
    )(branch_pool)

    return layers.Concatenate(axis=-1, name=f"{name}_output")(
        [branch_1_1, branch_3_3, branch_5_5, branch_pool]
    )


def build_inception(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout_rate: float = 0.5,
    train_images: np.ndarray | None = None,
    augment: bool = True,
    name: str = "inception",
) -> keras.Model:
    """Build the two-block inception network.

    :param train_images: if given, the normalization layer is adapted to these
        statistics; otherwise it passes the input through unchanged until
        someone calls ``adapt``.
    :param augment: include the random flip/rotation layers.  They are inert at
        inference time either way; turning them off keeps tests deterministic.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    inputs = keras.Input(shape=input_shape, name="input")

    x = build_normalization(train_images)(inputs)
    if augment:
        x = build_augmentation()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1_3_3")(x)
    x = layers.MaxPooling2D(2, name="pool1_3_3")(x)
    x = LocalResponseNormalization(name="lrn1")(x)

    x = inception_block(x, name="inception2")
    x = inception_block(x, name="inception3")

    x = layers.AveragePooling2D(7, strides=1, padding="same", name="pool4_7_7")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="target")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
