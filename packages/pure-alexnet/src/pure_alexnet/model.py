"""AlexNet-style classifier, ported from tflearn to Keras 3.

The original module built the network *and* downloaded the data-set *and*
trained *and* saved, all at import time, which made every part of it
untestable.  This module now only describes the architecture: it builds a
``keras.Model`` and hands it back.  Training lives in :mod:`pure_alexnet.train`.

The layer names are the ones the tflearn version used, so introspection
scripts written against the old graph still find their way around.
"""

from __future__ import annotations

import keras
from keras import layers

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (227, 227, 3)
DEFAULT_NUM_CLASSES = 17
DEFAULT_DENSE_UNITS = 2048


def build_alexnet(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = 0.5,
    name: str = "alexnet",
) -> keras.Model:
    """Build the AlexNet variant this project trains on 17 flower categories.

    ``dense_units`` is a parameter rather than a hard-coded 2048 so tests can
    build a structurally identical but small model; at the default input size
    the two 2048-unit dense layers alone are ~220M parameters.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    inputs = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(96, 7, strides=1, padding="same", activation="relu", name="conv1_7_7")(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="max_pool1_3_3_2")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2_3_3")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="max_pool2_3_3_2")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3_3_3")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv4_3_3")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="max_pool3_3_3_2")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="fully_connected_1_relu")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1_05")(x)
    x = layers.Dense(dense_units, activation="relu", name="fully_connected_2_relu")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2_05")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="fully_connected_3_softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def compile_alexnet(
    model: keras.Model,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
) -> keras.Model:
    """Attach the optimizer/loss the tflearn ``regression`` layer used to add.

    tflearn's ``optimizer='momentum'`` was SGD with momentum 0.9, and
    ``loss='categorical_crossentropy'`` expects one-hot labels - which is what
    :mod:`pure_alexnet.dataset` produces.
    """
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_and_compile(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
) -> keras.Model:
    """Convenience wrapper: :func:`build_alexnet` then :func:`compile_alexnet`."""
    model = build_alexnet(
        input_shape=input_shape,
        num_classes=num_classes,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )
    return compile_alexnet(model, learning_rate=learning_rate, momentum=momentum)
