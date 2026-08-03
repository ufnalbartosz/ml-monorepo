"""Tests for the pickle cache.

Both CNN packages depend on the "build once, reload instantly" contract, so it
is pinned here rather than twice in their suites.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_core.cache import (
    ensure_directories,
    load_or_build,
    read_pickle,
    write_pickle,
)


class TestReadWrite:
    def test_round_trip_preserves_arrays(self, tmp_path, rng):
        payload = {"images": rng.random((3, 4, 4, 3)), "labels": np.eye(3)}
        path = tmp_path / "data.pickle"

        write_pickle(payload, path)
        restored = read_pickle(path)

        np.testing.assert_array_equal(payload["images"], restored["images"])
        np.testing.assert_array_equal(payload["labels"], restored["labels"])

    def test_write_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.pickle"

        write_pickle({"x": 1}, path)

        assert path.exists()

    def test_write_returns_the_path(self, tmp_path):
        path = tmp_path / "data.pickle"

        assert write_pickle({"x": 1}, path) == path

    def test_write_overwrites_an_existing_file(self, tmp_path):
        path = tmp_path / "data.pickle"
        path.write_bytes(b"stale")

        write_pickle({"fresh": True}, path)

        assert read_pickle(path) == {"fresh": True}

    def test_read_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_pickle(tmp_path / "absent.pickle")


class TestLoadOrBuild:
    def test_builds_on_a_miss(self, tmp_path):
        path = tmp_path / "data.pickle"

        result = load_or_build(path, lambda: {"built": True})

        assert result == {"built": True}
        assert path.exists()

    def test_reuses_the_cache_on_a_hit(self, tmp_path):
        path = tmp_path / "data.pickle"
        write_pickle({"cached": True}, path)

        def explode():
            raise AssertionError("build must not run when the cache exists")

        assert load_or_build(path, explode) == {"cached": True}

    def test_builds_exactly_once_across_two_calls(self, tmp_path):
        path = tmp_path / "data.pickle"
        calls = []

        def build():
            calls.append(1)
            return {"n": len(calls)}

        first = load_or_build(path, build)
        second = load_or_build(path, build)

        assert len(calls) == 1
        assert first == second

    def test_creates_the_parent_directory_when_building(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "data.pickle"

        load_or_build(path, lambda: {"x": 1})

        assert path.exists()


class TestEnsureDirectories:
    def test_creates_a_missing_directory(self, tmp_path):
        target = tmp_path / "logs"

        ensure_directories(target)

        assert target.is_dir()

    def test_is_idempotent(self, tmp_path):
        target = tmp_path / "logs"

        ensure_directories(target)
        ensure_directories(target)

        assert target.is_dir()

    def test_creates_several_at_once(self, tmp_path):
        ensure_directories(tmp_path / "a", tmp_path / "b", tmp_path / "c")

        assert all((tmp_path / name).is_dir() for name in "abc")

    def test_creates_intermediate_directories(self, tmp_path):
        ensure_directories(tmp_path / "a" / "b" / "c")

        assert (tmp_path / "a" / "b" / "c").is_dir()

    def test_accepts_strings(self, tmp_path):
        ensure_directories(str(tmp_path / "from_string"))

        assert (tmp_path / "from_string").is_dir()
