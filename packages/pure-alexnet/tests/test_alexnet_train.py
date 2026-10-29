"""Tests for the training entrypoint.

The point of splitting ``train.py`` out of ``model.py`` was that the pipeline
could be exercised end to end on synthetic data; that is what
``test_main_runs_the_whole_pipeline`` does, in about a second.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest

from pure_alexnet import oxflower17
from pure_alexnet import train as train_module
from pure_alexnet.dataset import split_dataset
from pure_alexnet.model import build_and_compile

SMALL_INPUT_SHAPE = (16, 16, 3)
SMALL_NUM_CLASSES = 3


@pytest.fixture
def tiny_data(fake_raw_data):
    images, labels = fake_raw_data(num_classes=SMALL_NUM_CLASSES, per_class=10, size=16)
    return split_dataset(images, labels)


@pytest.fixture
def tiny_model():
    return build_and_compile(
        input_shape=SMALL_INPUT_SHAPE,
        num_classes=SMALL_NUM_CLASSES,
        dense_units=8,
    )


class TestAccuracy:
    def test_perfect_predictions_score_one(self):
        labels = np.eye(3, dtype=np.float32)
        assert train_module.accuracy(labels, labels) == pytest.approx(1.0)

    def test_completely_wrong_predictions_score_zero(self):
        labels = np.eye(3, dtype=np.float32)
        predictions = labels[::-1].copy()
        predictions[1] = [0.0, 0.0, 1.0]  # make every row disagree

        assert train_module.accuracy(predictions, labels) == pytest.approx(0.0)

    def test_counts_the_argmax_not_the_probability(self):
        labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        predictions = np.array([[0.51, 0.49], [0.9, 0.1]], dtype=np.float32)

        assert train_module.accuracy(predictions, labels) == pytest.approx(0.5)

    def test_rejects_an_empty_batch(self):
        with pytest.raises(ValueError, match="zero predictions"):
            train_module.accuracy(np.zeros((0, 3)), np.zeros((0, 3)))


class TestTrain:
    def test_runs_the_requested_number_of_epochs(self, tiny_model, tiny_data):
        history = train_module.train(tiny_model, tiny_data, epochs=2, batch_size=4, verbose=0)

        assert len(history.history["loss"]) == 2

    def test_reports_validation_metrics(self, tiny_model, tiny_data):
        history = train_module.train(tiny_model, tiny_data, epochs=1, batch_size=4, verbose=0)

        assert "val_loss" in history.history

    def test_callbacks_are_invoked(self, tiny_model, tiny_data):
        seen = []

        class Spy(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                seen.append(epoch)

        train_module.train(
            tiny_model, tiny_data, epochs=2, batch_size=4, callbacks=[Spy()], verbose=0
        )

        assert seen == [0, 1]


class TestEvaluate:
    def test_returns_a_fraction(self, tiny_model, tiny_data):
        score = train_module.evaluate(tiny_model, tiny_data)

        assert 0.0 <= score <= 1.0

    def test_agrees_with_a_hand_computed_accuracy(self, tiny_model, tiny_data):
        predictions = tiny_model.predict(tiny_data["test_images"], verbose=0)
        expected = train_module.accuracy(predictions, tiny_data["test_labels"])

        assert train_module.evaluate(tiny_model, tiny_data) == pytest.approx(expected)


class TestBuildCallbacks:
    def test_creates_the_directories_it_writes_to(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        train_module.build_callbacks(checkpoint_dir, log_dir)

        assert checkpoint_dir.is_dir()
        assert log_dir.is_dir()

    def test_checkpoint_path_carries_the_model_name(self, tmp_path):
        callbacks = train_module.build_callbacks(tmp_path / "ckpt", tmp_path / "logs", "3_3")

        checkpoint = next(c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint))
        assert checkpoint.filepath.endswith("3_3.keras")


class TestParseArgs:
    def test_defaults_match_the_original_script(self):
        args = train_module.parse_args([])

        assert args.epochs == 150
        assert args.batch_size == 64
        assert args.learning_rate == pytest.approx(0.001)
        assert args.model_name == "3_3"

    def test_overrides_are_applied(self):
        args = train_module.parse_args(["--epochs", "3", "--batch-size", "8"])

        assert args.epochs == 3
        assert args.batch_size == 8


def test_main_runs_the_whole_pipeline(tmp_path, monkeypatch, fake_raw_data):
    """Data loading -> build -> train -> evaluate -> save, on synthetic data."""
    images, labels = fake_raw_data(num_classes=SMALL_NUM_CLASSES, per_class=10, size=16)

    # DataSet resolves its default loader from this module attribute, so the
    # download never happens.
    monkeypatch.setattr(oxflower17, "load_data", lambda root, image_size: (images, labels))
    monkeypatch.chdir(tmp_path)

    exit_code = train_module.main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--dense-units",
            "8",
            "--dataset-path",
            str(tmp_path / "data" / "dataset.pickle"),
            "--save-dir",
            str(tmp_path / "saves"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "saves" / "3_3.keras").exists()
