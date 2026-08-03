"""Tests for image discovery and decoding."""

from __future__ import annotations

import numpy as np

from vision_core.images import image_paths, load_image, load_images


class TestImagePaths:
    def test_sorts_by_filename(self, tmp_path, write_jpeg):
        for name in ("image_0003.jpg", "image_0001.jpg", "image_0002.jpg"):
            write_jpeg(tmp_path / name)

        paths = image_paths(tmp_path)

        assert [p.name for p in paths] == [
            "image_0001.jpg",
            "image_0002.jpg",
            "image_0003.jpg",
        ]

    def test_ignores_files_that_are_not_images(self, tmp_path, write_jpeg):
        write_jpeg(tmp_path / "image_0001.jpg")
        (tmp_path / "notes.txt").write_text("not an image")

        assert [p.name for p in image_paths(tmp_path)] == ["image_0001.jpg"]

    def test_extension_filter_is_case_insensitive(self, tmp_path, write_jpeg):
        write_jpeg(tmp_path / "UPPER.JPG")

        assert len(image_paths(tmp_path)) == 1

    def test_extensions_can_be_narrowed(self, tmp_path, write_jpeg):
        write_jpeg(tmp_path / "a.jpg")
        write_jpeg(tmp_path / "b.png")

        paths = image_paths(tmp_path, extensions=(".png",))

        assert [p.name for p in paths] == ["b.png"]

    def test_empty_directory_yields_nothing(self, tmp_path):
        assert image_paths(tmp_path) == []


class TestLoadImage:
    def test_decodes_to_float32_in_the_unit_range(self, tmp_path, write_jpeg):
        path = write_jpeg(tmp_path / "red.jpg", size=(8, 8), colour=(255, 0, 0))

        image = load_image(path, image_size=(4, 4))

        assert image.shape == (4, 4, 3)
        assert image.dtype == np.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_preserves_colour(self, tmp_path, write_jpeg):
        path = write_jpeg(tmp_path / "red.jpg", size=(8, 8), colour=(255, 0, 0))

        image = load_image(path, image_size=(4, 4))

        assert image[0, 0, 0] > 0.9, "a pure red pixel should stay red"
        assert image[0, 0, 1] < 0.1

    def test_image_size_is_height_then_width(self, tmp_path, write_jpeg):
        """PIL takes (width, height); the array convention is the other way round."""
        path = write_jpeg(tmp_path / "wide.jpg", size=(40, 10))

        image = load_image(path, image_size=(6, 12))

        assert image.shape == (6, 12, 3)

    def test_greyscale_images_become_three_channel(self, tmp_path):
        from PIL import Image

        path = tmp_path / "grey.jpg"
        Image.new("L", (8, 8), 128).save(path, format="JPEG")

        assert load_image(path, image_size=(4, 4)).shape == (4, 4, 3)


class TestLoadImages:
    def test_stacks_into_a_batch(self, tmp_path, write_jpeg):
        paths = [
            write_jpeg(tmp_path / "a.jpg", size=(8, 12)),
            write_jpeg(tmp_path / "b.jpg", size=(20, 5)),
        ]

        images = load_images(paths, image_size=(6, 6))

        assert images.shape == (2, 6, 6, 3)

    def test_resizes_everything_to_a_common_shape(self, tmp_path, write_jpeg):
        paths = [write_jpeg(tmp_path / f"{i}.jpg", size=(i * 4 + 4, 8)) for i in range(3)]

        images = load_images(paths, image_size=(5, 5))

        assert images.shape == (3, 5, 5, 3)

    def test_empty_input_yields_an_empty_batch(self):
        assert load_images([], image_size=(4, 4)).shape == (0, 4, 4, 3)

    def test_preserves_input_order(self, tmp_path, write_jpeg):
        red = write_jpeg(tmp_path / "red.jpg", colour=(255, 0, 0))
        blue = write_jpeg(tmp_path / "blue.jpg", colour=(0, 0, 255))

        images = load_images([red, blue], image_size=(4, 4))

        assert images[0, 0, 0, 0] > images[1, 0, 0, 0]
