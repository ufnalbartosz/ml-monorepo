"""Tests for the AlexNet architecture.

These are the checks that were impossible while the model was built at import
time, next to a 150-epoch ``model.fit`` call.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest

from pure_alexnet.model import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    build_alexnet,
    build_and_compile,
    compile_alexnet,
)

# Structurally identical to the real model, small enough to fit in a test run.
SMALL_INPUT_SHAPE = (32, 32, 3)
SMALL_NUM_CLASSES = 4
SMALL_DENSE_UNITS = 8


@pytest.fixture
def small_model() -> keras.Model:
    return build_alexnet(
        input_shape=SMALL_INPUT_SHAPE,
        num_classes=SMALL_NUM_CLASSES,
        dense_units=SMALL_DENSE_UNITS,
    )


def test_output_shape_matches_num_classes(small_model):
    assert small_model.output_shape == (None, SMALL_NUM_CLASSES)


def test_input_shape_is_respected(small_model):
    assert small_model.input_shape == (None, *SMALL_INPUT_SHAPE)


def test_defaults_match_the_original_tflearn_graph():
    # The tflearn version hard-coded 227x227x3 -> 17 classes.
    assert DEFAULT_INPUT_SHAPE == (227, 227, 3)
    assert DEFAULT_NUM_CLASSES == 17


def test_layer_names_are_preserved_from_the_tflearn_version(small_model):
    names = [layer.name for layer in small_model.layers]

    assert names == [
        "input",
        "conv1_7_7",
        "max_pool1_3_3_2",
        "conv2_3_3",
        "max_pool2_3_3_2",
        "conv3_3_3",
        "conv4_3_3",
        "max_pool3_3_3_2",
        "flatten",
        "fully_connected_1_relu",
        "dropout_1_05",
        "fully_connected_2_relu",
        "dropout_2_05",
        "fully_connected_3_softmax",
    ]


def test_convolution_filter_counts(small_model):
    assert small_model.get_layer("conv1_7_7").filters == 96
    assert small_model.get_layer("conv1_7_7").kernel_size == (7, 7)

    for name in ("conv2_3_3", "conv3_3_3", "conv4_3_3"):
        layer = small_model.get_layer(name)
        assert layer.filters == 128
        assert layer.kernel_size == (3, 3)


def test_pooling_halves_the_spatial_dimensions(small_model):
    # 'same' padding with stride 2: 32 -> 16 -> 8 -> 4.
    assert tuple(small_model.get_layer("max_pool1_3_3_2").output.shape[1:3]) == (16, 16)
    assert tuple(small_model.get_layer("max_pool2_3_3_2").output.shape[1:3]) == (8, 8)
    assert tuple(small_model.get_layer("max_pool3_3_3_2").output.shape[1:3]) == (4, 4)


def test_predictions_are_a_probability_distribution(small_model, rng):
    images = rng.random((5, *SMALL_INPUT_SHAPE), dtype=np.float32)

    predictions = small_model.predict(images, verbose=0)

    assert predictions.shape == (5, SMALL_NUM_CLASSES)
    assert np.all(predictions >= 0.0)
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, rtol=1e-5)


def test_inference_is_deterministic_but_training_passes_are_not(small_model, rng):
    """Dropout must be active in training and inert at inference."""
    images = rng.random((4, *SMALL_INPUT_SHAPE), dtype=np.float32)

    first = np.asarray(small_model(images, training=False))
    second = np.asarray(small_model(images, training=False))
    np.testing.assert_allclose(first, second, rtol=1e-6)

    keras.utils.set_random_seed(0)
    train_a = np.asarray(small_model(images, training=True))
    keras.utils.set_random_seed(1)
    train_b = np.asarray(small_model(images, training=True))
    assert not np.allclose(train_a, train_b)


@pytest.mark.parametrize("num_classes", [0, -1])
def test_rejects_impossible_class_counts(num_classes):
    with pytest.raises(ValueError, match="num_classes"):
        build_alexnet(input_shape=SMALL_INPUT_SHAPE, num_classes=num_classes)


@pytest.mark.parametrize("dropout_rate", [-0.1, 1.0, 1.5])
def test_rejects_impossible_dropout_rates(dropout_rate):
    with pytest.raises(ValueError, match="dropout_rate"):
        build_alexnet(input_shape=SMALL_INPUT_SHAPE, dropout_rate=dropout_rate)


def test_compile_attaches_sgd_with_momentum(small_model):
    compile_alexnet(small_model, learning_rate=0.01, momentum=0.9)

    assert isinstance(small_model.optimizer, keras.optimizers.SGD)
    assert float(small_model.optimizer.learning_rate) == pytest.approx(0.01)
    assert float(small_model.optimizer.momentum) == pytest.approx(0.9)


def test_the_model_can_overfit_a_single_batch(rng):
    """A network that cannot drive the loss down on four images is broken.

    The seed is fixed and the optimizer is Adam on purpose: the question is
    whether the architecture can fit anything at all, not whether momentum SGD
    at one particular learning rate escapes one particular initialization.
    Dropout is off for the same reason - 50% dropout on a deliberately narrow
    dense layer is a test of the regularizer, not of the network.
    """
    keras.utils.set_random_seed(0)

    model = build_alexnet(
        input_shape=SMALL_INPUT_SHAPE,
        num_classes=SMALL_NUM_CLASSES,
        dense_units=32,
        dropout_rate=0.0,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
    )

    images = rng.random((4, *SMALL_INPUT_SHAPE), dtype=np.float32)
    labels = np.eye(SMALL_NUM_CLASSES, dtype=np.float32)

    history = model.fit(images, labels, epochs=30, batch_size=4, verbose=0)
    losses = history.history["loss"]

    assert losses[-1] < losses[0]


def test_build_and_compile_returns_a_trainable_model(rng):
    """The convenience wrapper must produce something `fit` accepts."""
    model = build_and_compile(
        input_shape=SMALL_INPUT_SHAPE,
        num_classes=SMALL_NUM_CLASSES,
        dense_units=SMALL_DENSE_UNITS,
    )

    images = rng.random((4, *SMALL_INPUT_SHAPE), dtype=np.float32)
    labels = np.eye(SMALL_NUM_CLASSES, dtype=np.float32)

    history = model.fit(images, labels, epochs=1, batch_size=4, verbose=0)

    assert "loss" in history.history


def test_model_survives_a_save_load_round_trip(tmp_path, small_model, rng):
    images = rng.random((3, *SMALL_INPUT_SHAPE), dtype=np.float32)
    before = small_model.predict(images, verbose=0)

    path = tmp_path / "alexnet.keras"
    small_model.save(path)
    reloaded = keras.models.load_model(path)

    np.testing.assert_allclose(before, reloaded.predict(images, verbose=0), rtol=1e-5)
