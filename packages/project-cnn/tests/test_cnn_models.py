"""Tests for the three architectures.

All three used to be built at import time next to a multi-hundred-epoch
``model.fit``, so none of this could be checked.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf

from project_cnn.alexnet_model import build_alexnet
from project_cnn.dual_path import build_dual_path
from project_cnn.inception import build_inception, inception_block
from project_cnn.layers import LocalResponseNormalization

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 4


def small_alexnet(**kwargs):
    return build_alexnet(
        **{"input_shape": INPUT_SHAPE, "num_classes": NUM_CLASSES, "dense_units": 32, **kwargs}
    )


def small_inception(**kwargs):
    return build_inception(
        **{"input_shape": INPUT_SHAPE, "num_classes": NUM_CLASSES, "augment": False, **kwargs}
    )


def small_dual_path(**kwargs):
    return build_dual_path(
        **{
            "input_shape": INPUT_SHAPE,
            "num_classes": NUM_CLASSES,
            "dense_units": (16, 8),
            **kwargs,
        }
    )


BUILDERS = {
    "alexnet": small_alexnet,
    "inception": small_inception,
    "dual_path": small_dual_path,
}


@pytest.mark.parametrize("builder", BUILDERS.values(), ids=BUILDERS.keys())
class TestEveryArchitecture:
    """Contracts that all three models must satisfy."""

    def test_output_is_one_probability_per_class(self, builder, rng):
        model = builder()
        images = rng.random((3, *INPUT_SHAPE), dtype=np.float32)

        predictions = model.predict(images, verbose=0)

        assert predictions.shape == (3, NUM_CLASSES)
        assert np.all(predictions >= 0.0)
        np.testing.assert_allclose(predictions.sum(axis=1), 1.0, rtol=1e-5)

    def test_input_shape_is_respected(self, builder):
        assert builder().input_shape == (None, *INPUT_SHAPE)

    def test_has_trainable_weights(self, builder):
        assert builder().trainable_weights

    def test_gradients_reach_every_trainable_weight(self, builder, rng):
        """No layer may be cut off from the loss - that is what a dead branch looks like."""
        model = builder()
        images = tf.constant(rng.random((4, *INPUT_SHAPE), dtype=np.float32))
        labels = tf.constant(np.eye(NUM_CLASSES, dtype=np.float32))

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = keras.losses.categorical_crossentropy(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_weights)

        for weight, gradient in zip(model.trainable_weights, gradients, strict=True):
            assert gradient is not None, f"{weight.path} receives no gradient"
            assert float(tf.reduce_max(tf.abs(gradient))) > 0.0, f"{weight.path} gradient is zero"

    def test_survives_a_save_load_round_trip(self, builder, tmp_path, rng):
        model = builder()
        images = rng.random((2, *INPUT_SHAPE), dtype=np.float32)
        before = model.predict(images, verbose=0)

        path = tmp_path / f"{model.name}.keras"
        model.save(path)
        reloaded = keras.models.load_model(path)

        np.testing.assert_allclose(before, reloaded.predict(images, verbose=0), rtol=1e-5)

    def test_rejects_impossible_class_counts(self, builder):
        with pytest.raises(ValueError, match="num_classes"):
            builder(num_classes=0)


@pytest.mark.parametrize(
    "builder",
    [small_alexnet, small_inception],
    ids=["alexnet", "inception"],
)
def test_deterministic_models_can_overfit_a_single_batch(builder, rng):
    """A model whose loss will not move on four images is structurally broken.

    Only the two models without stochastic pre-processing are checked here.
    dual_path randomly crops and colour-jitters every training pass by design,
    so whether it memorises four images in a handful of epochs is a property of
    the optimizer, not of the architecture; the gradient-flow test above covers
    it instead.

    The seed is fixed and the optimizer is Adam on purpose: the question is
    whether the architecture can fit anything at all, not whether plain SGD at
    a particular learning rate happens to escape AlexNet's tanh saturation from
    a particular random initialization.  Dropout is off for the same reason -
    50% dropout on a deliberately narrow dense layer is a test of the
    regularizer, not of the network.
    """
    keras.utils.set_random_seed(0)

    model = builder(dropout_rate=0.0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
    )

    images = rng.random((4, *INPUT_SHAPE), dtype=np.float32)
    labels = np.eye(NUM_CLASSES, dtype=np.float32)

    history = model.fit(images, labels, epochs=30, batch_size=4, verbose=0)

    assert history.history["loss"][-1] < history.history["loss"][0]


class TestAlexNet:
    def test_layer_order(self):
        names = [layer.name for layer in small_alexnet().layers]

        assert names == [
            "input",
            "conv1",
            "pool1",
            "lrn1",
            "conv2",
            "pool2",
            "lrn2",
            "conv3",
            "conv4",
            "conv5",
            "pool3",
            "lrn3",
            "flatten",
            "fc1",
            "dropout1",
            "fc2",
            "dropout2",
            "target",
        ]

    def test_filter_counts_match_the_paper(self):
        model = small_alexnet()

        assert [
            model.get_layer(n).filters for n in ("conv1", "conv2", "conv3", "conv4", "conv5")
        ] == [
            96,
            256,
            384,
            384,
            256,
        ]

    def test_uses_tanh_in_the_dense_layers_like_the_tflearn_version(self):
        model = small_alexnet()

        assert model.get_layer("fc1").activation is keras.activations.tanh
        assert model.get_layer("fc2").activation is keras.activations.tanh

    def test_three_pooling_stages_quarter_then_quarter_again(self):
        model = small_alexnet()

        # 32 -> 16 -> 8 -> 4 with stride-2 'same' pooling.
        assert tuple(model.get_layer("pool1").output.shape[1:3]) == (16, 16)
        assert tuple(model.get_layer("pool2").output.shape[1:3]) == (8, 8)
        assert tuple(model.get_layer("pool3").output.shape[1:3]) == (4, 4)

    @pytest.mark.parametrize("dropout_rate", [-0.1, 1.0])
    def test_rejects_impossible_dropout_rates(self, dropout_rate):
        with pytest.raises(ValueError, match="dropout_rate"):
            small_alexnet(dropout_rate=dropout_rate)


class TestInception:
    def test_block_concatenates_four_branches_on_the_channel_axis(self):
        inputs = keras.Input(shape=(8, 8, 16))

        output = inception_block(inputs, name="test", reduce_filters=64, branch_filters=32)

        # 64 (1x1) + 32 (3x3) + 32 (5x5) + 32 (pool 1x1) = 160.
        assert output.shape[-1] == 160

    def test_block_preserves_the_spatial_size(self):
        inputs = keras.Input(shape=(8, 8, 16))

        output = inception_block(inputs, name="test")

        assert tuple(output.shape[1:3]) == (8, 8)

    def test_branch_layer_names_carry_over_from_tflearn(self):
        names = {layer.name for layer in small_inception().layers}

        for expected in (
            "inception2a_1_1",
            "inception2b_3_3_reduce",
            "inception2b_3_3",
            "inception2c_5_5_reduce",
            "inception2c_5_5",
            "inception2d_pool",
            "inception2d_pool_1_1",
            "inception3a_1_1",
            "inception3d_pool_1_1",
        ):
            assert expected in names

    def test_normalization_adapts_to_the_training_statistics(self, rng):
        train_images = rng.normal(loc=5.0, scale=2.0, size=(64, *INPUT_SHAPE)).astype(np.float32)

        model = build_inception(
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            augment=False,
            train_images=train_images,
        )

        normalization = model.get_layer("normalization")
        normalized = np.asarray(normalization(train_images))
        assert abs(float(normalized.mean())) < 0.1
        assert abs(float(normalized.std()) - 1.0) < 0.1

    def test_augmentation_is_inert_at_inference(self, rng):
        model = build_inception(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, augment=True)
        images = rng.random((4, *INPUT_SHAPE), dtype=np.float32)

        first = np.asarray(model(images, training=False))
        second = np.asarray(model(images, training=False))

        np.testing.assert_allclose(first, second, rtol=1e-5)

    def test_augmentation_can_be_switched_off(self):
        names = {layer.name for layer in small_inception().layers}

        assert "augmentation" not in names


class TestDualPath:
    def test_pre_processing_crops_the_input(self):
        model = small_dual_path(img_size_cropped=24)

        assert tuple(model.get_layer("pre_process").output.shape[1:3]) == (24, 24)

    def test_first_subdivide_concatenates_both_branches(self):
        model = small_dual_path()

        # 64 from the 5x5 branch + 128 from the 3x3 branch.
        assert model.get_layer("inception_1_output").output.shape[-1] == 192

    def test_second_subdivide_halves_the_spatial_size(self):
        model = small_dual_path(img_size_cropped=24)

        output = model.get_layer("inception_2_output").output
        assert tuple(output.shape[1:3]) == (12, 12)
        assert output.shape[-1] == 96

    def test_batch_normalization_is_on_the_first_branch(self):
        model = small_dual_path()

        assert isinstance(model.get_layer("inception_1_0_bn"), keras.layers.BatchNormalization)

    def test_training_and_inference_share_one_set_of_weights(self, rng):
        """The TF1 version needed two graph copies under a reusing variable scope."""
        model = small_dual_path()
        images = rng.random((4, *INPUT_SHAPE), dtype=np.float32)

        model(images, training=True)
        model(images, training=False)

        weight_names = [w.path for w in model.weights]
        assert len(weight_names) == len(set(weight_names))

    def test_inference_is_deterministic_despite_the_random_crop(self, rng):
        model = small_dual_path()
        images = rng.random((4, *INPUT_SHAPE), dtype=np.float32)

        first = np.asarray(model(images, training=False))
        second = np.asarray(model(images, training=False))

        np.testing.assert_allclose(first, second, rtol=1e-5)

    def test_rejects_a_crop_larger_than_the_input(self):
        with pytest.raises(ValueError, match="img_size_cropped"):
            build_dual_path(input_shape=(16, 16, 3), img_size_cropped=24)


class TestLocalResponseNormalization:
    def test_preserves_shape(self, rng):
        layer = LocalResponseNormalization()
        inputs = rng.random((2, 8, 8, 16), dtype=np.float32)

        assert np.asarray(layer(inputs)).shape == (2, 8, 8, 16)

    def test_shrinks_large_activations(self, rng):
        layer = LocalResponseNormalization(alpha=1.0, beta=0.75)
        inputs = np.full((1, 4, 4, 8), 10.0, dtype=np.float32)

        output = np.asarray(layer(inputs))

        assert np.all(output < inputs)

    def test_leaves_zeros_alone(self):
        layer = LocalResponseNormalization()
        inputs = np.zeros((1, 4, 4, 8), dtype=np.float32)

        np.testing.assert_allclose(np.asarray(layer(inputs)), inputs)

    def test_config_round_trip(self):
        layer = LocalResponseNormalization(depth_radius=3, bias=2.0, alpha=0.01, beta=0.5)

        restored = LocalResponseNormalization.from_config(layer.get_config())

        assert restored.depth_radius == 3
        assert restored.bias == 2.0
        assert restored.alpha == 0.01
        assert restored.beta == 0.5
