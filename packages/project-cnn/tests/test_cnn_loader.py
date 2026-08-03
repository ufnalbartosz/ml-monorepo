"""Tests for the CIFAR-100 loader.

The Python 2 pickle handling is the interesting part: the archive on
cs.toronto.edu was written by Python 2, so on Python 3 every key comes back as
``bytes``.  ``test_unpickle_decodes_python2_byte_keys`` pins that down against
a fixture pickled the same way.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from project_cnn import loader
from project_cnn.loader import (
    Cifar100Config,
    convert_images,
    convert_labels,
    decode_bytes,
    load_class_names,
    load_data,
    subset_mask,
    unpickle,
)


@pytest.fixture
def cifar_dir(tmp_path, cifar_class_names):
    """A miniature CIFAR-100 tree with byte keys, as Python 2 would have written it."""
    extracted = tmp_path / "cifar-100-python"
    extracted.mkdir(parents=True)

    with (extracted / "meta").open("wb") as fp:
        pickle.dump({b"fine_label_names": [n.encode() for n in cifar_class_names]}, fp)

    # Six rows: two 'bottle', two 'apple', two of a class we do not keep.
    fine_labels = [0, 0, 1, 1, 50, 50]
    raw = np.arange(6 * 3 * 32 * 32, dtype=np.uint8).reshape(6, -1)
    with (extracted / "test").open("wb") as fp:
        pickle.dump({b"data": raw, b"fine_labels": fine_labels}, fp)

    return tmp_path


@pytest.fixture
def config(cifar_dir, cifar_class_names):
    # Keep the classes sitting at fine-labels 0 and 1.
    return Cifar100Config(
        data_path=cifar_dir,
        class_names=(cifar_class_names[0], cifar_class_names[1]),
    )


class TestDecodeBytes:
    def test_decodes_keys_and_string_values(self):
        assert decode_bytes({b"key": b"value"}) == {"key": "value"}

    def test_decodes_lists_of_strings(self):
        assert decode_bytes([b"a", b"b"]) == ["a", "b"]

    def test_leaves_numpy_arrays_untouched(self):
        array = np.arange(6, dtype=np.uint8)

        result = decode_bytes({b"data": array})

        assert result["data"] is array

    def test_leaves_integers_untouched(self):
        assert decode_bytes({b"fine_labels": [3, 7]}) == {"fine_labels": [3, 7]}


class TestUnpickle:
    def test_decodes_python2_byte_keys(self, config):
        """On Python 3 the raw pickle gives b'data'; the loader must give 'data'."""
        data = unpickle("test", config)

        assert "data" in data
        assert "fine_labels" in data

    def test_reads_from_the_configured_directory(self, tmp_path, config):
        assert config.extracted_dir == tmp_path / "cifar-100-python"

    def test_missing_file_raises(self, config):
        with pytest.raises(FileNotFoundError):
            unpickle("nonexistent", config)


class TestConvertImages:
    def test_reshapes_channel_first_rows_to_channel_last(self):
        raw = np.zeros((2, 3 * 32 * 32), dtype=np.uint8)

        images = convert_images(raw)

        assert images.shape == (2, 32, 32, 3)

    def test_scales_bytes_into_the_unit_range(self):
        raw = np.full((1, 3 * 4 * 4), 255, dtype=np.uint8)

        images = convert_images(raw, image_size=4, channels=3)

        np.testing.assert_allclose(images, 1.0)

    def test_channel_ordering_is_correct(self):
        # A row that is all-red: the first plane is 255, the other two are 0.
        raw = np.concatenate(
            [np.full(16, 255, np.uint8), np.zeros(16, np.uint8), np.zeros(16, np.uint8)]
        ).reshape(1, -1)

        images = convert_images(raw, image_size=4, channels=3)

        np.testing.assert_allclose(images[0, :, :, 0], 1.0)
        np.testing.assert_allclose(images[0, :, :, 1], 0.0)
        np.testing.assert_allclose(images[0, :, :, 2], 0.0)


class TestSubsetMask:
    def test_keeps_only_the_wanted_classes(self, cifar_class_names):
        fine_labels = np.array([0, 5, 1, 99])

        mask = subset_mask(fine_labels, cifar_class_names, [cifar_class_names[0]])

        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_returns_a_boolean_array(self, cifar_class_names):
        mask = subset_mask(np.array([0, 1]), cifar_class_names, [cifar_class_names[0]])

        assert mask.dtype == bool


class TestConvertLabels:
    def test_renumbers_into_the_kept_class_order(self, cifar_class_names):
        keep = [cifar_class_names[7], cifar_class_names[3]]
        fine_labels = np.array([7, 3, 7])

        converted, mask = convert_labels(fine_labels, cifar_class_names, keep)

        np.testing.assert_array_equal(converted, [0, 1, 0])
        assert mask.sum() == 3

    def test_drops_the_classes_outside_keep(self, cifar_class_names):
        keep = [cifar_class_names[0]]
        fine_labels = np.array([0, 42, 0])

        converted, mask = convert_labels(fine_labels, cifar_class_names, keep)

        assert len(converted) == 2
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_labels_stay_inside_the_valid_range(self, cifar_class_names):
        keep = cifar_class_names[:5]
        fine_labels = np.array([0, 1, 2, 3, 4, 90])

        converted, _ = convert_labels(fine_labels, cifar_class_names, keep)

        assert converted.min() >= 0
        assert converted.max() < len(keep)


class TestLoadData:
    def test_returns_images_class_numbers_and_one_hot_labels(self, config):
        images, cls, one_hot = load_data("test", config)

        assert images.shape == (4, 32, 32, 3)
        assert cls.shape == (4,)
        assert one_hot.shape == (4, 2)

    def test_drops_the_rows_outside_the_configured_classes(self, config):
        # The fixture holds six rows, two of which belong to a class not kept.
        _, cls, _ = load_data("test", config)

        assert len(cls) == 4

    def test_one_hot_labels_agree_with_the_class_numbers(self, config):
        _, cls, one_hot = load_data("test", config)

        np.testing.assert_array_equal(np.argmax(one_hot, axis=1), cls)

    def test_class_names_come_from_the_meta_file(self, config, cifar_class_names):
        assert load_class_names(config) == cifar_class_names


class TestMaybeDownloadAndExtract:
    def test_passes_the_configured_url_and_directory_through(self, tmp_path, monkeypatch):
        """Downloading itself is vision-core's job; this checks the wiring."""
        seen = {}

        def fake(url, download_dir):
            seen.update(url=url, download_dir=download_dir)
            return tmp_path / "archive.tar.gz"

        monkeypatch.setattr(loader, "download_and_extract", fake)
        config = Cifar100Config(data_path=tmp_path)

        loader.maybe_download_and_extract(config)

        assert seen["url"] == loader.DATA_URL
        assert seen["download_dir"] == tmp_path


class TestConfig:
    def test_num_classes_follows_the_class_name_list(self):
        assert Cifar100Config(class_names=("a", "b", "c")).num_classes == 3

    def test_defaults_keep_the_twenty_selected_classes(self):
        assert Cifar100Config().num_classes == 20
        assert loader.num_classes == 20

    def test_class_names_are_sorted(self):
        assert list(loader.labels) == sorted(loader.labels)
