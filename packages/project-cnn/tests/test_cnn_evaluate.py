"""Tests for batched prediction and accuracy reporting.

These functions used to read module-level graph handles, so they could only
run inside ``main.py``. They now take a model and arrays.
"""

from __future__ import annotations

import keras
import numpy as np
import pytest

from project_cnn.evaluate import (
    classification_accuracy,
    evaluate_split,
    predict_cls,
    print_test_accuracy,
    print_valid_accuracy,
)

NUM_CLASSES = 4


class ConstantModel(keras.Model):
    """A model that always predicts ``class_index`` - lets accuracy be exact."""

    def __init__(self, class_index: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.class_index = class_index
        self.num_classes = num_classes

    def call(self, inputs):
        batch = keras.ops.shape(inputs)[0]
        one_hot = keras.ops.one_hot(
            keras.ops.full((batch,), self.class_index, dtype="int32"), self.num_classes
        )
        return keras.ops.cast(one_hot, "float32")


@pytest.fixture
def constant_model():
    return ConstantModel(class_index=0)


class TestPredictCls:
    def test_returns_one_prediction_per_image(self, constant_model, rng):
        images = rng.random((7, 8, 8, 3), dtype=np.float32)
        cls_true = np.zeros(7, dtype=np.int64)

        correct, cls_pred = predict_cls(constant_model, images, cls_true, batch_size=3)

        assert cls_pred.shape == (7,)
        assert correct.shape == (7,)

    def test_batching_does_not_change_the_result(self, constant_model, rng):
        images = rng.random((10, 8, 8, 3), dtype=np.float32)
        cls_true = np.arange(10) % NUM_CLASSES

        _, one_batch = predict_cls(constant_model, images, cls_true, batch_size=100)
        _, many_batches = predict_cls(constant_model, images, cls_true, batch_size=3)

        np.testing.assert_array_equal(one_batch, many_batches)

    def test_marks_the_matching_rows_correct(self, constant_model, rng):
        images = rng.random((4, 8, 8, 3), dtype=np.float32)
        cls_true = np.array([0, 1, 0, 2], dtype=np.int64)

        correct, _ = predict_cls(constant_model, images, cls_true)

        np.testing.assert_array_equal(correct, [True, False, True, False])

    def test_a_batch_size_larger_than_the_input_still_works(self, constant_model, rng):
        images = rng.random((2, 8, 8, 3), dtype=np.float32)

        correct, _ = predict_cls(constant_model, images, np.zeros(2, np.int64), batch_size=999)

        assert len(correct) == 2

    def test_rejects_mismatched_lengths(self, constant_model, rng):
        images = rng.random((4, 8, 8, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="disagree on length"):
            predict_cls(constant_model, images, np.zeros(3, np.int64))


class TestClassificationAccuracy:
    def test_all_correct_scores_one(self):
        assert classification_accuracy(np.array([True, True])) == (1.0, 2)

    def test_none_correct_scores_zero(self):
        assert classification_accuracy(np.array([False, False])) == (0.0, 0)

    def test_half_correct(self):
        accuracy, count = classification_accuracy(np.array([True, False, True, False]))

        assert accuracy == pytest.approx(0.5)
        assert count == 2

    def test_rejects_an_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            classification_accuracy(np.array([], dtype=bool))


class TestEvaluateSplit:
    def test_accuracy_matches_a_hand_count(self, fake_dataset):
        dataset = fake_dataset(num_classes=NUM_CLASSES, per_split=12)
        model = ConstantModel(class_index=0)

        accuracy, correct, cls_pred = evaluate_split(model, dataset, "test")

        # cls cycles 0..3, so exactly a quarter of the rows are class 0.
        assert accuracy == pytest.approx(0.25)
        assert np.all(cls_pred == 0)
        assert correct.sum() == 3

    @pytest.mark.parametrize("split", ["train", "valid", "test"])
    def test_works_on_every_split(self, fake_dataset, split):
        dataset = fake_dataset()

        accuracy, _, _ = evaluate_split(ConstantModel(0), dataset, split)

        assert 0.0 <= accuracy <= 1.0


class TestPrinting:
    def test_valid_accuracy_is_reported(self, fake_dataset, capsys):
        accuracy = print_valid_accuracy(ConstantModel(0), fake_dataset())

        assert "Accuracy on Validation-Set" in capsys.readouterr().out
        assert accuracy == pytest.approx(0.25)

    def test_test_accuracy_is_reported(self, fake_dataset, capsys):
        accuracy = print_test_accuracy(ConstantModel(0), fake_dataset())

        assert "Accuracy on Test-Set" in capsys.readouterr().out
        assert accuracy == pytest.approx(0.25)

    def test_confusion_matrix_can_be_printed(self, fake_dataset, capsys):
        print_test_accuracy(ConstantModel(0), fake_dataset(), show_confusion_matrix=True)

        assert "Confusion Matrix" in capsys.readouterr().out

    def test_example_errors_are_written_next_to_the_working_directory(
        self, fake_dataset, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)

        print_test_accuracy(ConstantModel(0), fake_dataset(), show_example_errors=True)

        assert (tmp_path / "example_errors.png").exists()
