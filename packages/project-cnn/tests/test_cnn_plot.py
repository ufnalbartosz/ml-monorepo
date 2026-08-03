"""Tests for the plotting helpers.

They write PNGs through the Agg backend, so they run headless. Under TF1 three
of these functions needed a live ``tf.Session`` and the graph's placeholder
handles; they now take arrays and models.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest

from project_cnn import plot
from project_cnn.tools import PreProcessing, get_layer_output, get_weights_variable


@pytest.fixture
def toy_model():
    inputs = keras.Input(shape=(8, 8, 3), name="input")
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="layer_conv1")(inputs)
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(2, activation="softmax", name="target")(x)
    return keras.Model(inputs, outputs, name="toy")


class TestPlotImages:
    def test_writes_a_file(self, tmp_path, rng):
        images = rng.random((9, 8, 8, 3), dtype=np.float32)
        target = tmp_path / "images.png"

        plot.plot_images(
            images=images,
            cls_true=np.arange(9) % 3,
            class_names=["a", "b", "c"],
            filename=str(target),
        )

        assert target.exists()

    def test_handles_fewer_than_nine_images(self, tmp_path, rng):
        images = rng.random((2, 8, 8, 3), dtype=np.float32)
        target = tmp_path / "few.png"

        plot.plot_images(
            images=images,
            cls_true=np.array([0, 1]),
            class_names=["a", "b"],
            filename=str(target),
        )

        assert target.exists()

    def test_empty_input_writes_nothing(self, tmp_path, capsys):
        plot.plot_images(
            images=np.zeros((0, 8, 8, 3), dtype=np.float32),
            cls_true=np.array([], dtype=int),
            class_names=["a"],
            filename=str(tmp_path / "empty.png"),
        )

        assert "Nothing to plot" in capsys.readouterr().out
        assert not (tmp_path / "empty.png").exists()


class TestPlotExampleErrors:
    def test_writes_a_file_when_there_are_errors(self, tmp_path, monkeypatch, fake_dataset):
        monkeypatch.chdir(tmp_path)
        dataset = fake_dataset(num_classes=4, per_split=12)
        cls_pred = (dataset["test_cls"] + 1) % 4
        correct = np.zeros(len(cls_pred), dtype=bool)

        plot.plot_example_errors(cls_pred=cls_pred, correct=correct, dataset=dataset)

        assert (tmp_path / "example_errors.png").exists()

    def test_says_nothing_to_plot_when_everything_was_right(
        self, tmp_path, monkeypatch, fake_dataset, capsys
    ):
        monkeypatch.chdir(tmp_path)
        dataset = fake_dataset()
        correct = np.ones(len(dataset["test_cls"]), dtype=bool)

        plot.plot_example_errors(cls_pred=dataset["test_cls"], correct=correct, dataset=dataset)

        assert "No misclassified images" in capsys.readouterr().out


class TestPlotConfusionMatrix:
    def test_prints_one_row_per_class(self, fake_dataset, capsys):
        dataset = fake_dataset(num_classes=4)

        plot.plot_confusion_matrix(cls_pred=dataset["test_cls"], dataset=dataset)

        output = capsys.readouterr().out
        for name in dataset["class_names"]:
            assert name in output


class TestPlotConvWeights:
    def test_accepts_a_keras_kernel_without_a_session(self, tmp_path, toy_model):
        weights = get_weights_variable(toy_model, "layer_conv1")
        target = tmp_path / "weights.png"

        plot.plot_conv_weights(weights, filename=str(target))

        assert target.exists()

    def test_handles_a_single_filter(self, tmp_path, rng):
        # num_grids == 1, which used to break `axes.flat`.
        weights = rng.random((3, 3, 3, 1), dtype=np.float32)
        target = tmp_path / "one_filter.png"

        plot.plot_conv_weights(weights, filename=str(target))

        assert target.exists()


class TestPlotLayerOutput:
    def test_plots_the_activations_of_a_layer(self, tmp_path, toy_model, rng):
        activations = get_layer_output(toy_model, "layer_conv1")
        image = rng.random((8, 8, 3), dtype=np.float32)
        target = tmp_path / "layer_output.png"

        plot.plot_layer_output(activations, image, filename=str(target))

        assert target.exists()


class TestPlotDistortedImage:
    def test_plots_nine_distortions_of_one_image(self, tmp_path, rng, fake_dataset):
        dataset = fake_dataset()
        image = rng.random((32, 32, 3), dtype=np.float32)
        distort = PreProcessing(img_size_cropped=24, num_channels=3)
        target = tmp_path / "distorted.png"

        plot.plot_distorted_image(
            image=image,
            cls_true=0,
            distort=lambda batch: distort(batch, training=True),
            dataset=dataset,
            filename=str(target),
        )

        assert target.exists()


class TestPlotImage:
    def test_writes_a_file(self, tmp_path, rng):
        target = tmp_path / "image.png"

        plot.plot_image(rng.random((8, 8, 3), dtype=np.float32), filename=str(target))

        assert target.exists()
