"""Tests for the data-set loader that replaced ``tflearn.datasets.oxflower17``.

The download is injected, and the archive fixture is built on the fly, so
nothing here reaches the network.
"""

from __future__ import annotations

import tarfile

import numpy as np
import pytest
from PIL import Image

from pure_alexnet import oxflower17


def write_jpeg(path, size=(16, 16), colour=(255, 0, 0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="JPEG")
    return path


class TestClassNumbers:
    def test_first_block_of_images_is_the_first_class(self):
        numbers = oxflower17.class_numbers(240, images_per_class=80)

        assert numbers[0] == 0
        assert numbers[79] == 0
        assert numbers[80] == 1
        assert numbers[239] == 2

    def test_full_dataset_has_seventeen_balanced_classes(self):
        numbers = oxflower17.class_numbers(1360)

        counts = np.bincount(numbers)
        assert len(counts) == oxflower17.NUM_CLASSES
        assert set(counts) == {oxflower17.IMAGES_PER_CLASS}

    def test_rejects_a_non_positive_block_size(self):
        with pytest.raises(ValueError, match="images_per_class"):
            oxflower17.class_numbers(10, images_per_class=0)


class TestOneHotEncoded:
    def test_encodes_to_a_single_one_per_row(self):
        encoded = oxflower17.one_hot_encoded([0, 2, 1], num_classes=3)

        np.testing.assert_array_equal(
            encoded,
            np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
        )

    def test_matches_the_class_numbers_it_came_from(self):
        numbers = oxflower17.class_numbers(1360)

        encoded = oxflower17.one_hot_encoded(numbers, oxflower17.NUM_CLASSES)

        assert encoded.shape == (1360, 17)
        np.testing.assert_array_equal(np.argmax(encoded, axis=1), numbers)


class TestImagePaths:
    def test_returns_jpegs_sorted_by_name(self, tmp_path):
        for name in ("image_0003.jpg", "image_0001.jpg", "image_0002.jpg"):
            write_jpeg(tmp_path / name)

        paths = oxflower17.image_paths(tmp_path)

        assert [p.name for p in paths] == ["image_0001.jpg", "image_0002.jpg", "image_0003.jpg"]

    def test_ignores_non_jpeg_files(self, tmp_path):
        write_jpeg(tmp_path / "image_0001.jpg")
        (tmp_path / "files.txt").write_text("not an image")

        assert [p.name for p in oxflower17.image_paths(tmp_path)] == ["image_0001.jpg"]


class TestLoadImages:
    def test_decodes_to_float_in_the_unit_range(self, tmp_path):
        path = write_jpeg(tmp_path / "image_0001.jpg", size=(8, 8), colour=(255, 0, 0))

        image = oxflower17.load_image(path, image_size=(4, 4))

        assert image.shape == (4, 4, 3)
        assert image.dtype == np.float32
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert image[0, 0, 0] > 0.9, "a pure red pixel should stay red"

    def test_resizes_every_image_to_the_requested_shape(self, tmp_path):
        paths = [
            write_jpeg(tmp_path / "image_0001.jpg", size=(8, 12)),
            write_jpeg(tmp_path / "image_0002.jpg", size=(20, 5)),
        ]

        images = oxflower17.load_images(paths, image_size=(6, 6))

        assert images.shape == (2, 6, 6, 3)

    def test_empty_input_yields_an_empty_batch(self):
        assert oxflower17.load_images([], image_size=(4, 4)).shape == (0, 4, 4, 3)


class TestLoadData:
    @pytest.fixture
    def archive(self, tmp_path):
        """A miniature stand-in for 17flowers.tgz: 3 classes, 2 images each."""
        staging = tmp_path / "staging" / "jpg"
        for i in range(1, 7):
            write_jpeg(staging / f"image_{i:04d}.jpg", size=(10, 10))

        archive_path = tmp_path / "17flowers.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging, arcname="jpg")

        return archive_path

    def test_downloads_extracts_and_decodes(self, tmp_path, archive):
        root = tmp_path / "data"
        downloads = []

        def downloader(download_dir, **kwargs):
            downloads.append(download_dir)
            return archive

        images, labels = oxflower17.load_data(
            root=root,
            image_size=(8, 8),
            downloader=downloader,
        )

        assert downloads == [root]
        assert images.shape == (6, 8, 8, 3)
        assert labels.shape == (6, oxflower17.NUM_CLASSES)

    def test_skips_the_download_when_images_are_already_extracted(self, tmp_path):
        root = tmp_path / "data"
        for i in range(1, 4):
            write_jpeg(root / "jpg" / f"image_{i:04d}.jpg", size=(10, 10))

        def downloader(**kwargs):
            raise AssertionError("should not download when jpg/ already exists")

        images, _ = oxflower17.load_data(root=root, image_size=(8, 8), downloader=downloader)

        assert len(images) == 3

    def test_can_return_integer_labels(self, tmp_path, archive):
        _, labels = oxflower17.load_data(
            root=tmp_path / "data",
            image_size=(8, 8),
            one_hot=False,
            downloader=lambda download_dir, **kwargs: archive,
        )

        assert labels.ndim == 1

    def test_raises_when_the_archive_has_no_images(self, tmp_path):
        empty_staging = tmp_path / "staging" / "jpg"
        empty_staging.mkdir(parents=True)
        archive_path = tmp_path / "empty.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(empty_staging, arcname="jpg")

        with pytest.raises(FileNotFoundError, match="No JPEG files"):
            oxflower17.load_data(
                root=tmp_path / "data",
                downloader=lambda download_dir, **kwargs: archive_path,
            )


class TestExtract:
    def test_raises_when_the_tarball_has_no_jpg_directory(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "readme.txt").write_text("nothing useful")
        archive_path = tmp_path / "wrong.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging, arcname="other")

        with pytest.raises(FileNotFoundError, match="No 'jpg' directory"):
            oxflower17.extract(archive_path, tmp_path / "out")


def test_class_names_line_up_with_the_class_count():
    assert len(oxflower17.CLASS_NAMES) == oxflower17.NUM_CLASSES
    assert len(set(oxflower17.CLASS_NAMES)) == oxflower17.NUM_CLASSES


def test_download_skips_an_archive_that_is_already_on_disk(tmp_path):
    existing = tmp_path / "17flowers.tgz"
    existing.write_bytes(b"already here")

    result = oxflower17.download(url=oxflower17.DATA_URL, download_dir=tmp_path)

    assert result == existing
    assert existing.read_bytes() == b"already here"
