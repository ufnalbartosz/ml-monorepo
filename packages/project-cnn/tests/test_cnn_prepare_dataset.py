"""Tests for the split-building and caching layer."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from project_cnn import prepare_dataset
from project_cnn.prepare_dataset import (
    DATASET_KEYS,
    create_dataset,
    load_dataset,
    read_pickle,
    split_test_dataset,
    write_pickle,
)


@pytest.fixture
def test_split_arrays():
    """20 rows across 4 classes, 5 each - the shape CIFAR's test-set has."""
    num_classes = 4
    cls = np.repeat(np.arange(num_classes), 5).astype(np.int64)
    labels = np.eye(num_classes, dtype=np.float32)[cls]
    images = np.arange(len(cls) * 8 * 8 * 3, dtype=np.float32).reshape(len(cls), 8, 8, 3)
    return images, cls, labels


class TestSplitTestDataset:
    def test_produces_both_halves(self, test_split_arrays):
        result = split_test_dataset(*test_split_arrays)

        assert set(result) == {
            "test_images",
            "test_labels",
            "test_cls",
            "valid_images",
            "valid_labels",
            "valid_cls",
        }

    def test_every_row_lands_in_exactly_one_half(self, test_split_arrays):
        images, cls, labels = test_split_arrays

        result = split_test_dataset(images, cls, labels)

        assert len(result["test_cls"]) + len(result["valid_cls"]) == len(cls)

    def test_halves_are_disjoint(self, test_split_arrays):
        images, cls, labels = test_split_arrays

        result = split_test_dataset(images, cls, labels)

        # Row 0 of each image is a unique fingerprint here.
        test_ids = {float(img[0, 0, 0]) for img in result["test_images"]}
        valid_ids = {float(img[0, 0, 0]) for img in result["valid_images"]}
        assert test_ids.isdisjoint(valid_ids)

    def test_both_halves_keep_every_class(self, test_split_arrays):
        images, cls, labels = test_split_arrays

        result = split_test_dataset(images, cls, labels)

        assert set(result["test_cls"]) == set(cls)
        assert set(result["valid_cls"]) == set(cls)

    def test_takes_every_other_row_of_each_class_for_validation(self, test_split_arrays):
        images, cls, labels = test_split_arrays

        result = split_test_dataset(images, cls, labels)

        # 5 rows per class -> indices 0, 2, 4 go to validation.
        assert len(result["valid_cls"]) == 4 * 3
        assert len(result["test_cls"]) == 4 * 2

    def test_images_stay_aligned_with_their_labels(self, test_split_arrays):
        images, cls, labels = test_split_arrays

        result = split_test_dataset(images, cls, labels)

        np.testing.assert_array_equal(
            np.argmax(result["valid_labels"], axis=1), result["valid_cls"]
        )
        np.testing.assert_array_equal(np.argmax(result["test_labels"], axis=1), result["test_cls"])

    def test_split_size_follows_the_data_not_a_hard_coded_2000(self):
        """The pre-port code assumed a 2000-row test-set and broke when labels changed."""
        cls = np.repeat(np.arange(2), 4).astype(np.int64)
        labels = np.eye(2, dtype=np.float32)[cls]
        images = np.zeros((len(cls), 4, 4, 3), dtype=np.float32)

        result = split_test_dataset(images, cls, labels)

        assert len(result["valid_cls"]) + len(result["test_cls"]) == 8


class TestPickleRoundTrip:
    def test_write_then_read_returns_the_same_arrays(self, tmp_path, fake_dataset):
        dataset = fake_dataset()
        path = tmp_path / "nested" / "dataset.pickle"

        write_pickle(dataset, path)
        restored = read_pickle(path)

        for key in ("train_images", "test_cls", "valid_labels"):
            np.testing.assert_array_equal(dataset[key], restored[key])

    def test_write_creates_the_parent_directory(self, tmp_path, fake_dataset):
        path = tmp_path / "a" / "b" / "dataset.pickle"

        write_pickle(fake_dataset(), path)

        assert path.exists()

    def test_write_overwrites_an_existing_file(self, tmp_path, fake_dataset):
        path = tmp_path / "dataset.pickle"
        path.write_bytes(b"stale")

        write_pickle(fake_dataset(), path)

        with path.open("rb") as fp:
            assert isinstance(pickle.load(fp), dict)

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_pickle(tmp_path / "absent.pickle")


class TestLoadDataset:
    def test_returns_the_cache_when_it_exists(self, tmp_path, fake_dataset, monkeypatch):
        dataset = fake_dataset()
        path = tmp_path / "dataset.pickle"
        write_pickle(dataset, path)

        def explode(*args, **kwargs):
            raise AssertionError("should not rebuild when the cache exists")

        monkeypatch.setattr(prepare_dataset, "create_dataset", explode)

        restored = load_dataset(path)

        np.testing.assert_array_equal(restored["train_images"], dataset["train_images"])

    def test_builds_the_cache_when_it_is_missing(self, tmp_path, fake_dataset, monkeypatch):
        dataset = fake_dataset()
        calls = []

        def fake_create(dataset_path, config):
            calls.append(dataset_path)
            return dataset

        monkeypatch.setattr(prepare_dataset, "create_dataset", fake_create)

        result = load_dataset(tmp_path / "missing.pickle")

        assert calls == [tmp_path / "missing.pickle"]
        assert result is dataset


class TestCreateDataset:
    def test_assembles_all_the_expected_keys(self, tmp_path, monkeypatch, test_split_arrays):
        images, cls, labels = test_split_arrays

        monkeypatch.setattr(prepare_dataset.loader, "maybe_download_and_extract", lambda c: None)
        monkeypatch.setattr(
            prepare_dataset.loader, "load_data", lambda name, config: (images, cls, labels)
        )

        path = tmp_path / "dataset.pickle"
        dataset = create_dataset(path)

        assert set(dataset) == set(DATASET_KEYS)
        assert path.exists()

    def test_class_names_come_from_the_config(self, tmp_path, monkeypatch, test_split_arrays):
        images, cls, labels = test_split_arrays

        monkeypatch.setattr(prepare_dataset.loader, "maybe_download_and_extract", lambda c: None)
        monkeypatch.setattr(
            prepare_dataset.loader, "load_data", lambda name, config: (images, cls, labels)
        )

        dataset = create_dataset(tmp_path / "dataset.pickle")

        assert dataset["class_names"] == list(prepare_dataset.DEFAULT_CONFIG.class_names)
