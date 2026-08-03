"""Tests for pre-processing and model introspection.

Under TF1 none of this could be checked without standing up a session and a
graph; the ops now run eagerly.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from project_cnn.tools import (
    PreProcessing,
    get_layer_output,
    get_weights_variable,
    pre_process,
    pre_process_image,
)

IMG_SIZE = 32
CROPPED = 24
CHANNELS = 3


@pytest.fixture
def image(rng):
    return tf.constant(rng.random((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.float32))


@pytest.fixture
def batch(rng):
    return tf.constant(rng.random((5, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.float32))


class TestPreProcessImage:
    def test_training_crops_to_the_requested_size(self, image):
        result = pre_process_image(image, True, CROPPED, CHANNELS)

        assert tuple(result.shape) == (CROPPED, CROPPED, CHANNELS)

    def test_evaluation_crops_to_the_requested_size(self, image):
        result = pre_process_image(image, False, CROPPED, CHANNELS)

        assert tuple(result.shape) == (CROPPED, CROPPED, CHANNELS)

    def test_training_output_stays_inside_the_unit_range(self, image):
        # The colour jitter can overflow; the clip must put it back.
        for _ in range(5):
            result = np.asarray(pre_process_image(image, True, CROPPED, CHANNELS))
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_evaluation_is_deterministic(self, image):
        first = np.asarray(pre_process_image(image, False, CROPPED, CHANNELS))
        second = np.asarray(pre_process_image(image, False, CROPPED, CHANNELS))

        np.testing.assert_array_equal(first, second)

    def test_evaluation_takes_the_centre_crop(self):
        # A frame of ones around a block of zeros: a centre crop keeps zeros.
        array = np.ones((8, 8, 1), dtype=np.float32)
        array[2:6, 2:6, :] = 0.0

        result = np.asarray(pre_process_image(tf.constant(array), False, 4, 1))

        np.testing.assert_allclose(result, 0.0)

    def test_training_distorts_differently_each_call(self, image):
        first = np.asarray(pre_process_image(image, True, CROPPED, CHANNELS))
        second = np.asarray(pre_process_image(image, True, CROPPED, CHANNELS))

        assert not np.allclose(first, second)

    def test_evaluation_pads_an_image_smaller_than_the_crop(self):
        small = tf.constant(np.ones((4, 4, 3), dtype=np.float32))

        result = pre_process_image(small, False, 8, 3)

        assert tuple(result.shape) == (8, 8, 3)


class TestPreProcess:
    def test_maps_over_the_whole_batch(self, batch):
        result = pre_process(batch, True, CROPPED, CHANNELS)

        assert tuple(result.shape) == (5, CROPPED, CROPPED, CHANNELS)

    def test_each_image_gets_its_own_distortion(self):
        # One image repeated: independent random crops must diverge.
        single = np.zeros((1, 16, 16, 3), dtype=np.float32)
        single[0, 4:8, 4:8, :] = 1.0
        repeated = tf.constant(np.repeat(single, 8, axis=0))

        result = np.asarray(pre_process(repeated, True, 8, 3))

        assert not all(np.allclose(result[0], result[i]) for i in range(1, 8))

    def test_evaluation_leaves_the_batch_deterministic(self, batch):
        first = np.asarray(pre_process(batch, False, CROPPED, CHANNELS))
        second = np.asarray(pre_process(batch, False, CROPPED, CHANNELS))

        np.testing.assert_array_equal(first, second)


class TestPreProcessingLayer:
    def test_output_shape_is_the_cropped_size(self, batch):
        layer = PreProcessing(img_size_cropped=CROPPED, num_channels=CHANNELS)

        assert tuple(layer(batch, training=False).shape) == (5, CROPPED, CROPPED, CHANNELS)

    def test_compute_output_shape_agrees_with_the_real_output(self, batch):
        layer = PreProcessing(img_size_cropped=CROPPED, num_channels=CHANNELS)

        declared = layer.compute_output_shape((None, IMG_SIZE, IMG_SIZE, CHANNELS))
        actual = layer(batch, training=False).shape

        assert declared[1:] == tuple(actual[1:])

    def test_training_flag_switches_between_distortion_and_centre_crop(self, batch):
        layer = PreProcessing(img_size_cropped=CROPPED, num_channels=CHANNELS)

        deterministic = np.asarray(layer(batch, training=False))
        again = np.asarray(layer(batch, training=False))
        distorted = np.asarray(layer(batch, training=True))

        np.testing.assert_array_equal(deterministic, again)
        assert not np.allclose(deterministic, distorted)

    def test_defaults_to_the_evaluation_path(self, batch):
        layer = PreProcessing(img_size_cropped=CROPPED, num_channels=CHANNELS)

        np.testing.assert_array_equal(
            np.asarray(layer(batch)), np.asarray(layer(batch, training=False))
        )

    def test_config_round_trip(self):
        layer = PreProcessing(img_size_cropped=12, num_channels=1)

        restored = PreProcessing.from_config(layer.get_config())

        assert restored.img_size_cropped == 12
        assert restored.num_channels == 1


@pytest.fixture
def toy_model():
    inputs = keras.Input(shape=(8, 8, 3), name="input")
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="layer_conv1")(inputs)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="target")(x)
    return keras.Model(inputs, outputs, name="toy")


class TestGetWeightsVariable:
    def test_returns_the_convolution_kernel(self, toy_model):
        weights = get_weights_variable(toy_model, "layer_conv1")

        assert tuple(weights.shape) == (3, 3, 3, 4)

    def test_the_kernel_is_the_live_variable(self, toy_model):
        weights = get_weights_variable(toy_model, "layer_conv1")

        assert weights is toy_model.get_layer("layer_conv1").kernel

    def test_raises_for_a_layer_without_weights(self, toy_model):
        with pytest.raises(ValueError, match="no kernel"):
            get_weights_variable(toy_model, "flatten")

    def test_raises_for_an_unknown_layer(self, toy_model):
        with pytest.raises(ValueError):
            get_weights_variable(toy_model, "does_not_exist")


class TestGetLayerOutput:
    def test_returns_the_activations_of_that_layer(self, toy_model, rng):
        activations = get_layer_output(toy_model, "layer_conv1")
        images = rng.random((2, 8, 8, 3), dtype=np.float32)

        result = activations.predict(images, verbose=0)

        assert result.shape == (2, 8, 8, 4)

    def test_activations_agree_with_calling_the_layer_directly(self, toy_model, rng):
        images = rng.random((2, 8, 8, 3), dtype=np.float32)
        activations = get_layer_output(toy_model, "layer_conv1")

        direct = np.asarray(toy_model.get_layer("layer_conv1")(images))

        np.testing.assert_allclose(activations.predict(images, verbose=0), direct, rtol=1e-5)

    def test_does_not_assume_relu_like_the_tf1_tensor_name_lookup_did(self, rng):
        inputs = keras.Input(shape=(4, 4, 1), name="input")
        outputs = keras.layers.Conv2D(2, 3, padding="same", activation="tanh", name="c")(inputs)
        model = keras.Model(inputs, outputs)

        activations = get_layer_output(model, "c")

        assert activations.predict(rng.random((1, 4, 4, 1), dtype=np.float32), verbose=0).shape == (
            1,
            4,
            4,
            2,
        )
