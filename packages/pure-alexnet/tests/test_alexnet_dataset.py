"""Tests for the splitting logic and the on-disk cache.

None of these touch the network: :class:`DataSet` takes its loader as a
constructor argument precisely so this file can hand it random arrays.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pure_alexnet.dataset import (
    LABELS,
    SPLIT_KEYS,
    DataSet,
    ensure_directories,
    split_dataset,
    train_test_valid_split,
)


def one_hot(class_numbers, num_classes):
    return np.eye(num_classes, dtype=np.float32)[np.asarray(class_numbers)]


class TestTrainTestValidSplit:
    def test_splits_are_disjoint_and_complete(self, fake_raw_data):
        _, labels = fake_raw_data(num_classes=4, per_class=10)

        splits = train_test_valid_split(labels)

        all_indices = splits.train + splits.test + splits.valid
        assert sorted(all_indices) == list(range(len(labels)))
        assert len(set(all_indices)) == len(all_indices)

    def test_each_class_contributes_the_same_number_of_rows(self, fake_raw_data):
        _, labels = fake_raw_data(num_classes=4, per_class=10)

        splits = train_test_valid_split(labels, test_fraction=0.2, valid_fraction=0.1)

        # 10 per class: 2 test, 1 validation, 7 train.
        assert len(splits.test) == 4 * 2
        assert len(splits.valid) == 4 * 1
        assert len(splits.train) == 4 * 7

        for indices, expected_per_class in ((splits.test, 2), (splits.valid, 1)):
            classes = np.argmax(labels[indices], axis=1)
            counts = np.bincount(classes, minlength=4)
            assert list(counts) == [expected_per_class] * 4

    def test_the_real_dataset_shape_reproduces_the_original_counts(self):
        # 1360 images, 17 classes, 80 per class -> 16 test / 8 valid per class.
        labels = one_hot(np.arange(1360) // 80, 17)

        splits = train_test_valid_split(labels)

        assert len(splits.test) == 17 * 16
        assert len(splits.valid) == 17 * 8
        assert len(splits.train) == 17 * 56

    def test_unpacks_as_a_triple(self, fake_raw_data):
        _, labels = fake_raw_data()

        train, test, valid = train_test_valid_split(labels)

        assert len(train) + len(test) + len(valid) == len(labels)

    def test_rejects_non_one_hot_input(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            train_test_valid_split(np.arange(10))

    def test_rejects_fractions_that_exceed_the_data(self, fake_raw_data):
        _, labels = fake_raw_data()

        with pytest.raises(ValueError, match="must be in"):
            train_test_valid_split(labels, test_fraction=0.8, valid_fraction=0.5)


class TestSplitDataset:
    def test_produces_all_six_arrays(self, fake_raw_data):
        images, labels = fake_raw_data()

        dataset = split_dataset(images, labels)

        assert set(dataset) == set(SPLIT_KEYS)

    def test_images_and_labels_stay_aligned(self, fake_raw_data):
        images, labels = fake_raw_data(num_classes=3, per_class=10)

        dataset = split_dataset(images, labels)

        for split in ("train", "test", "valid"):
            assert len(dataset[f"{split}_images"]) == len(dataset[f"{split}_labels"])

        # Each row must still be paired with the label it started with.
        for image, label in zip(dataset["test_images"], dataset["test_labels"], strict=True):
            row = np.flatnonzero((images == image).all(axis=(1, 2, 3)))
            assert len(row) == 1
            np.testing.assert_array_equal(labels[row[0]], label)

    def test_no_image_appears_in_two_splits(self, fake_raw_data):
        images, labels = fake_raw_data()

        dataset = split_dataset(images, labels)

        total = sum(len(dataset[f"{s}_images"]) for s in ("train", "test", "valid"))
        assert total == len(images)

    def test_rejects_mismatched_lengths(self, fake_raw_data):
        images, labels = fake_raw_data()

        with pytest.raises(ValueError, match="disagree on length"):
            split_dataset(images[:-1], labels)


class TestDataSet:
    def test_builds_the_pickle_on_first_load(self, tmp_path, fake_raw_data):
        images, labels = fake_raw_data()
        calls = []

        def loader(root, image_size):
            calls.append((root, image_size))
            return images, labels

        dataset_path = tmp_path / "17flowers" / "dataset.pickle"
        data = DataSet(dataset_path=dataset_path, loader=loader).load()

        assert len(calls) == 1
        assert dataset_path.exists()
        assert set(data) == set(SPLIT_KEYS)

    def test_second_load_reads_the_cache_instead_of_the_loader(self, tmp_path, fake_raw_data):
        images, labels = fake_raw_data()
        calls = []

        def loader(root, image_size):
            calls.append(root)
            return images, labels

        dataset_path = tmp_path / "dataset.pickle"
        first = DataSet(dataset_path=dataset_path, loader=loader).load()
        second = DataSet(dataset_path=dataset_path, loader=loader).load()

        assert len(calls) == 1, "the cached pickle should have been reused"
        for key in SPLIT_KEYS:
            np.testing.assert_array_equal(first[key], second[key])

    def test_cached_pickle_holds_the_six_arrays(self, tmp_path, fake_raw_data):
        images, labels = fake_raw_data()
        dataset_path = tmp_path / "dataset.pickle"

        DataSet(dataset_path=dataset_path, loader=lambda root, image_size: (images, labels)).load()

        with dataset_path.open("rb") as fp:
            written = pickle.load(fp)
        assert set(written) == set(SPLIT_KEYS)

    def test_loader_is_passed_the_configured_root_and_image_size(self, tmp_path, fake_raw_data):
        images, labels = fake_raw_data()
        seen = {}

        def loader(root, image_size):
            seen.update(root=root, image_size=image_size)
            return images, labels

        root = tmp_path / "flowers"
        DataSet(dataset_path=root / "dataset.pickle", loader=loader, image_size=(64, 64)).load()

        assert seen["root"] == root
        assert seen["image_size"] == (64, 64)

    def test_cleanup_removes_the_raw_archive_and_images(self, tmp_path, fake_raw_data):
        images, labels = fake_raw_data()
        root = tmp_path / "17flowers"
        jpg_dir = root / "jpg"
        jpg_dir.mkdir(parents=True)
        (jpg_dir / "image_0001.jpg").write_bytes(b"not really a jpeg")
        (root / "17flowers.tgz").write_bytes(b"not really a tarball")

        DataSet(
            dataset_path=root / "dataset.pickle",
            loader=lambda root, image_size: (images, labels),
        ).load()

        assert not jpg_dir.exists()
        assert not (root / "17flowers.tgz").exists()
        assert (root / "dataset.pickle").exists()

    def test_cleanup_is_safe_when_nothing_was_downloaded(self, tmp_path):
        DataSet(dataset_path=tmp_path / "dataset.pickle").cleanup()

    def test_root_is_the_parent_of_the_pickle(self, tmp_path):
        dataset = DataSet(dataset_path=tmp_path / "sub" / "dataset.pickle")

        assert dataset.root == tmp_path / "sub"


def test_labels_cover_every_class():
    assert len(LABELS) == 17
    assert len(set(LABELS)) == 17


def test_ensure_directories_is_idempotent(tmp_path):
    target = tmp_path / "logs"

    ensure_directories(target)
    ensure_directories(target)

    assert target.is_dir()
