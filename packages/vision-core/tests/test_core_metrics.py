"""Tests for the accuracy helpers."""

from __future__ import annotations

import keras
import numpy as np
import pytest

from vision_core.labels import one_hot_encoded
from vision_core.metrics import (
    accuracy_from_probabilities,
    classification_accuracy,
    correct_predictions,
    predict_classes,
)


class ConstantModel(keras.Model):
    """Always predicts ``class_index``, so expected accuracy is exact."""

    def __init__(self, class_index: int, num_classes: int = 4):
        super().__init__()
        self.class_index = class_index
        self.num_classes = num_classes

    def call(self, inputs):
        batch = keras.ops.shape(inputs)[0]
        one_hot = keras.ops.one_hot(
            keras.ops.full((batch,), self.class_index, dtype="int32"), self.num_classes
        )
        return keras.ops.cast(one_hot, "float32")


class TestOneHotEncoded:
    def test_puts_a_single_one_per_row(self):
        np.testing.assert_array_equal(
            one_hot_encoded([0, 2, 1], num_classes=3),
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
        )

    def test_infers_the_class_count_when_omitted(self):
        assert one_hot_encoded([0, 1, 2]).shape == (3, 3)

    def test_accepts_a_wider_class_count_than_the_data_uses(self):
        assert one_hot_encoded([0, 1], num_classes=10).shape == (2, 10)

    def test_argmax_recovers_the_input(self):
        numbers = np.array([3, 0, 7, 7])

        encoded = one_hot_encoded(numbers, num_classes=8)

        np.testing.assert_array_equal(np.argmax(encoded, axis=1), numbers)


class TestPredictClasses:
    def test_returns_one_class_per_image(self, rng):
        images = rng.random((7, 8, 8, 3), dtype=np.float32)

        assert predict_classes(ConstantModel(0), images, batch_size=3).shape == (7,)

    def test_batching_does_not_change_the_result(self, rng):
        images = rng.random((10, 8, 8, 3), dtype=np.float32)
        model = ConstantModel(2)

        one_batch = predict_classes(model, images, batch_size=100)
        many = predict_classes(model, images, batch_size=3)

        np.testing.assert_array_equal(one_batch, many)

    def test_a_batch_size_larger_than_the_input_still_works(self, rng):
        images = rng.random((2, 8, 8, 3), dtype=np.float32)

        assert len(predict_classes(ConstantModel(0), images, batch_size=999)) == 2

    def test_returns_the_predicted_class(self, rng):
        images = rng.random((3, 8, 8, 3), dtype=np.float32)

        assert set(predict_classes(ConstantModel(3), images)) == {3}


class TestCorrectPredictions:
    def test_marks_matching_rows(self):
        np.testing.assert_array_equal(
            correct_predictions(np.array([0, 1, 0]), np.array([0, 2, 0])),
            [True, False, True],
        )

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="disagree on length"):
            correct_predictions(np.array([0, 1]), np.array([0]))


class TestClassificationAccuracy:
    def test_all_correct(self):
        assert classification_accuracy(np.array([True, True])) == (1.0, 2)

    def test_none_correct(self):
        assert classification_accuracy(np.array([False, False])) == (0.0, 0)

    def test_half_correct(self):
        accuracy, count = classification_accuracy(np.array([True, False, True, False]))

        assert accuracy == pytest.approx(0.5)
        assert count == 2

    def test_rejects_an_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            classification_accuracy(np.array([], dtype=bool))


class TestAccuracyFromProbabilities:
    def test_perfect_predictions_score_one(self):
        labels = np.eye(3, dtype=np.float32)

        assert accuracy_from_probabilities(labels, labels) == pytest.approx(1.0)

    def test_counts_the_argmax_not_the_margin(self):
        labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        predictions = np.array([[0.51, 0.49], [0.9, 0.1]], dtype=np.float32)

        assert accuracy_from_probabilities(predictions, labels) == pytest.approx(0.5)

    def test_completely_wrong_predictions_score_zero(self):
        labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        predictions = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        assert accuracy_from_probabilities(predictions, labels) == pytest.approx(0.0)

    def test_rejects_an_empty_batch(self):
        with pytest.raises(ValueError, match="zero predictions"):
            accuracy_from_probabilities(np.zeros((0, 3)), np.zeros((0, 3)))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="disagree on length"):
            accuracy_from_probabilities(np.eye(3), np.eye(2))

    def test_agrees_with_the_integer_path(self, rng):
        """The two scoring routes must not drift apart."""
        images = rng.random((12, 8, 8, 3), dtype=np.float32)
        cls_true = np.arange(12) % 4
        model = ConstantModel(1)

        via_integers, _ = classification_accuracy(
            correct_predictions(predict_classes(model, images), cls_true)
        )
        via_probabilities = accuracy_from_probabilities(
            model.predict(images, verbose=0), one_hot_encoded(cls_true, 4)
        )

        assert via_integers == pytest.approx(via_probabilities)
