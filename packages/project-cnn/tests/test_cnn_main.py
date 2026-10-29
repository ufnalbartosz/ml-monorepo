"""Tests for the CLI.

``main.py`` used to run the whole experiment as an import side effect. It is
now a function that takes argv, which is what makes
``test_main_runs_the_whole_pipeline`` possible.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest

from project_cnn import main as main_module
from project_cnn.main import (
    MODEL_BUILDERS,
    build_callbacks,
    build_model,
    compile_model,
    parse_args,
    restore_if_available,
    train,
)

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 4


@pytest.fixture
def tiny_model():
    model = build_model("dual-path", INPUT_SHAPE, NUM_CLASSES, dense_units=(16, 8))
    return compile_model(model, learning_rate=0.01)


class TestBuildModel:
    @pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
    def test_every_registered_name_builds(self, name):
        model = build_model(name, INPUT_SHAPE, NUM_CLASSES)

        assert model.output_shape == (None, NUM_CLASSES)

    def test_unknown_name_lists_the_valid_options(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("resnet", INPUT_SHAPE, NUM_CLASSES)

    def test_extra_keyword_arguments_reach_the_builder(self):
        model = build_model("alexnet", INPUT_SHAPE, NUM_CLASSES, dense_units=8)

        assert model.get_layer("fc1").units == 8


class TestCompileModel:
    def test_uses_plain_sgd_like_the_tf1_optimizer(self, tiny_model):
        assert isinstance(tiny_model.optimizer, keras.optimizers.SGD)

    def test_learning_rate_is_applied(self):
        model = compile_model(build_model("alexnet", INPUT_SHAPE, NUM_CLASSES, dense_units=4), 1e-4)

        assert float(model.optimizer.learning_rate) == pytest.approx(1e-4)


class TestTrain:
    def test_runs_the_requested_epochs(self, tiny_model, fake_dataset):
        history = train(tiny_model, fake_dataset(NUM_CLASSES), epochs=2, batch_size=4, verbose=0)

        assert len(history.history["loss"]) == 2

    def test_monitors_the_validation_split(self, tiny_model, fake_dataset):
        history = train(tiny_model, fake_dataset(NUM_CLASSES), epochs=1, batch_size=4, verbose=0)

        assert "val_loss" in history.history

    def test_does_not_touch_the_test_split(self, tiny_model, fake_dataset):
        """Training must never see the test-set; picking on it would leak it."""
        dataset = fake_dataset(NUM_CLASSES)
        before = dataset["test_images"].copy()

        train(tiny_model, dataset, epochs=1, batch_size=4, verbose=0)

        np.testing.assert_array_equal(dataset["test_images"], before)


class TestCallbacksAndCheckpoints:
    def test_build_callbacks_creates_the_save_directory(self, tmp_path):
        save_dir = tmp_path / "logs"

        build_callbacks(save_dir)

        assert save_dir.is_dir()

    def test_restore_returns_the_fresh_model_when_no_checkpoint_exists(self, tmp_path, tiny_model):
        restored = restore_if_available(tiny_model, tmp_path)

        assert restored is tiny_model

    def test_restore_loads_a_saved_checkpoint(self, tmp_path, tiny_model, rng):
        tiny_model.save(tmp_path / "model.keras")
        images = rng.random((2, *INPUT_SHAPE), dtype=np.float32)
        expected = tiny_model.predict(images, verbose=0)

        restored = restore_if_available(
            build_model("dual-path", INPUT_SHAPE, NUM_CLASSES, dense_units=(16, 8)), tmp_path
        )

        np.testing.assert_allclose(restored.predict(images, verbose=0), expected, rtol=1e-5)

    def test_restore_falls_back_when_the_checkpoint_is_corrupt(self, tmp_path, tiny_model, capsys):
        (tmp_path / "model.keras").write_bytes(b"this is not a keras archive")

        restored = restore_if_available(tiny_model, tmp_path)

        assert restored is tiny_model
        assert "Failed to restore checkpoint" in capsys.readouterr().out


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])

        assert args.model == "dual-path"
        assert args.learning_rate == pytest.approx(1e-4)
        assert args.resume is False

    def test_model_choice_is_validated(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "resnet"])

    def test_flags_are_parsed(self):
        args = parse_args(["--model", "inception", "--epochs", "3", "--resume"])

        assert args.model == "inception"
        assert args.epochs == 3
        assert args.resume is True


@pytest.mark.parametrize("model_name", sorted(MODEL_BUILDERS))
def test_main_runs_the_whole_pipeline(tmp_path, monkeypatch, fake_dataset, model_name):
    """Load -> build -> compile -> train -> evaluate, on synthetic data."""
    dataset = fake_dataset(num_classes=NUM_CLASSES, per_split=8)
    monkeypatch.setattr(main_module, "load_dataset", lambda path: dataset)
    monkeypatch.chdir(tmp_path)

    exit_code = main_module.main(
        [
            "--model",
            model_name,
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--save-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "logs" / "model.keras").exists()
