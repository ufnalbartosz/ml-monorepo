"""AlexNet applied to the 20-class subset of CIFAR-100 selected in loader.py.

Ported from tflearn to Keras 3.  The module used to build the graph, download
the data-set and run 100 epochs at import time; it now only describes the
architecture.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
      Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - CIFAR-100 Dataset. Alex Krizhevsky.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
"""

from __future__ import annotations

import keras
from keras import layers

from project_cnn.layers import LocalResponseNormalization

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (32, 32, 3)
DEFAULT_NUM_CLASSES = 20
DEFAULT_DENSE_UNITS = 4096


def build_alexnet(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = 0.5,
    name: str = "alexnet",
) -> keras.Model:
    """Build AlexNet for 32x32 CIFAR images.

    ``dense_units`` is an argument rather than a hard-coded 4096 so that tests
    can build the same structure at a size that fits in a test run.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0, 1)")

    inputs = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(96, 3, strides=1, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)
    x = LocalResponseNormalization(name="lrn1")(x)

    x = layers.Conv2D(256, 5, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool2")(x)
    x = LocalResponseNormalization(name="lrn2")(x)

    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv5")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool3")(x)
    x = LocalResponseNormalization(name="lrn3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="tanh", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(dense_units, activation="tanh", name="fc2")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="target")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)
