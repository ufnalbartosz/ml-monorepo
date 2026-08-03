"""Tests for the shared training loop, callbacks and checkpoint restore."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from vision_core.training import build_callbacks, restore_if_available, train

INPUT_SHAPE = (8, 8, 3)
NUM_CLASSES = 3


@pytest.fixture
def tiny_model():
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class TestBuildCallbacks:
    def test_creates_the_directories_it_writes_to(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "logs"

        build_callbacks(checkpoint_dir, log_dir)

        assert checkpoint_dir.is_dir()
        assert log_dir.is_dir()

    def test_logs_next_to_the_checkpoints_when_no_log_dir_is_given(self, tmp_path):
        callbacks = build_callbacks(tmp_path / "run")

        tensorboard = next(c for c in callbacks if isinstance(c, keras.callbacks.TensorBoard))
        assert str(tmp_path / "run") in str(tensorboard.log_dir)

    def test_checkpoint_path_carries_the_model_name(self, tmp_path):
        callbacks = build_callbacks(tmp_path / "ckpt", model_name="3_3")

        checkpoint = next(c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint))
        assert checkpoint.filepath.endswith("3_3.keras")

    def test_keeps_only_the_best_snapshot(self, tmp_path):
        callbacks = build_callbacks(tmp_path / "ckpt")

        checkpoint = next(c for c in callbacks if isinstance(c, keras.callbacks.ModelCheckpoint))
        assert checkpoint.save_best_only is True
        assert checkpoint.monitor == "val_accuracy"

    def test_writes_a_checkpoint_during_a_real_fit(self, tmp_path, tiny_model, fake_split_dataset):
        callbacks = build_callbacks(tmp_path / "ckpt", model_name="m")

        train(tiny_model, fake_split_dataset(NUM_CLASSES), 1, 4, callbacks=callbacks, verbose=0)

        assert (tmp_path / "ckpt" / "m.keras").exists()


class TestTrain:
    def test_runs_the_requested_epochs(self, tiny_model, fake_split_dataset):
        history = train(tiny_model, fake_split_dataset(NUM_CLASSES), 2, 4, verbose=0)

        assert len(history.history["loss"]) == 2

    def test_reports_validation_metrics(self, tiny_model, fake_split_dataset):
        history = train(tiny_model, fake_split_dataset(NUM_CLASSES), 1, 4, verbose=0)

        assert "val_loss" in history.history

    def test_never_reads_the_test_split(self, tiny_model, fake_split_dataset):
        """Selecting on the test-set leaks it into model selection."""
        dataset = fake_split_dataset(NUM_CLASSES)
        del dataset["test_images"], dataset["test_labels"], dataset["test_cls"]

        train(tiny_model, dataset, 1, 4, verbose=0)

    def test_callbacks_are_invoked(self, tiny_model, fake_split_dataset):
        seen = []

        class Spy(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                seen.append(epoch)

        train(tiny_model, fake_split_dataset(NUM_CLASSES), 2, 4, callbacks=[Spy()], verbose=0)

        assert seen == [0, 1]

    @pytest.mark.parametrize(
        "missing", ["train_images", "train_labels", "valid_images", "valid_labels"]
    )
    def test_names_the_key_it_is_missing(self, tiny_model, fake_split_dataset, missing):
        dataset = fake_split_dataset(NUM_CLASSES)
        del dataset[missing]

        with pytest.raises(KeyError, match=missing):
            train(tiny_model, dataset, 1, 4, verbose=0)


class TestRestoreIfAvailable:
    def test_returns_the_fresh_model_when_nothing_is_saved(self, tmp_path, tiny_model):
        assert restore_if_available(tiny_model, tmp_path) is tiny_model

    def test_loads_a_saved_model(self, tmp_path, tiny_model, rng):
        tiny_model.save(tmp_path / "model.keras")
        images = rng.random((2, *INPUT_SHAPE), dtype=np.float32)
        expected = tiny_model.predict(images, verbose=0)

        restored = restore_if_available(keras.Sequential([keras.Input(INPUT_SHAPE)]), tmp_path)

        np.testing.assert_allclose(restored.predict(images, verbose=0), expected, rtol=1e-5)

    def test_honours_the_model_name(self, tmp_path, tiny_model):
        tiny_model.save(tmp_path / "custom.keras")

        restored = restore_if_available(tiny_model, tmp_path, model_name="custom")

        assert restored is not tiny_model

    def test_falls_back_when_the_checkpoint_is_corrupt(self, tmp_path, tiny_model, capsys):
        (tmp_path / "model.keras").write_bytes(b"this is not a keras archive")

        restored = restore_if_available(tiny_model, tmp_path)

        assert restored is tiny_model
        assert "Failed to restore checkpoint" in capsys.readouterr().out
