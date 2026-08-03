"""Tests for the data-set loader that replaced ``tflearn.datasets.oxflower17``.

Only what is specific to this data-set is here: the class names, and the rule
that an image's position in the sorted file listing implies its label.  Generic
downloading, extraction and JPEG decoding are tested in vision-core.

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
            oxflower17.extract_archive(archive_path, tmp_path / "out")


def test_class_names_line_up_with_the_class_count():
    assert len(oxflower17.CLASS_NAMES) == oxflower17.NUM_CLASSES
    assert len(set(oxflower17.CLASS_NAMES)) == oxflower17.NUM_CLASSES


def test_download_archive_targets_the_vgg_url(tmp_path):
    existing = tmp_path / "17flowers.tgz"
    existing.write_bytes(b"already here")

    result = oxflower17.download_archive(download_dir=tmp_path)

    assert result == existing, "the archive name must match the configured URL"
    assert existing.read_bytes() == b"already here"
