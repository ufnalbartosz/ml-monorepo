"""Tests for archive download and extraction.

Nothing here reaches the network: ``urlretrieve`` is monkeypatched and the
tarballs are built in ``tmp_path``.
"""

from __future__ import annotations

import tarfile
import zipfile

import pytest

from vision_core import archives
from vision_core.archives import (
    download,
    download_and_extract,
    extract,
    print_download_progress,
)


@pytest.fixture
def payload_dir(tmp_path):
    directory = tmp_path / "payload"
    directory.mkdir()
    (directory / "marker.txt").write_text("hello")
    return directory


@pytest.fixture
def tarball(tmp_path, payload_dir):
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(payload_dir, arcname="payload")
    return path


@pytest.fixture
def fake_download(monkeypatch):
    """Make urlretrieve copy a local file instead of fetching a URL."""

    def _install(source):
        def fake_urlretrieve(url, filename, reporthook=None):
            filename.write_bytes(source.read_bytes())
            return filename, None

        monkeypatch.setattr(archives.urllib.request, "urlretrieve", fake_urlretrieve)

    return _install


class TestDownload:
    def test_skips_a_file_that_is_already_there(self, tmp_path):
        existing = tmp_path / "archive.tar.gz"
        existing.write_bytes(b"already here")

        result = download("https://example.invalid/archive.tar.gz", tmp_path)

        assert result == existing
        assert existing.read_bytes() == b"already here"

    def test_fetches_into_the_requested_directory(self, tmp_path, tarball, fake_download):
        fake_download(tarball)
        dest = tmp_path / "downloads"

        result = download("https://example.invalid/archive.tar.gz", dest)

        assert result == dest / "archive.tar.gz"
        assert result.exists()

    def test_creates_the_download_directory(self, tmp_path, tarball, fake_download):
        fake_download(tarball)
        dest = tmp_path / "a" / "b"

        download("https://example.invalid/archive.tar.gz", dest)

        assert dest.is_dir()

    def test_names_the_file_after_the_url(self, tmp_path, tarball, fake_download):
        fake_download(tarball)

        result = download("https://example.invalid/some/path/cifar-100.tar.gz", tmp_path)

        assert result.name == "cifar-100.tar.gz"


class TestExtract:
    def test_unpacks_a_tarball(self, tmp_path, tarball):
        dest = tmp_path / "out"

        extract(tarball, dest)

        assert (dest / "payload" / "marker.txt").read_text() == "hello"

    def test_unpacks_a_zip(self, tmp_path, payload_dir):
        archive_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.write(payload_dir / "marker.txt", arcname="payload/marker.txt")
        dest = tmp_path / "out"

        extract(archive_path, dest)

        assert (dest / "payload" / "marker.txt").read_text() == "hello"

    def test_returns_the_destination(self, tmp_path, tarball):
        dest = tmp_path / "out"

        assert extract(tarball, dest) == dest

    def test_creates_the_destination(self, tmp_path, tarball):
        dest = tmp_path / "deep" / "out"

        extract(tarball, dest)

        assert dest.is_dir()

    def test_rejects_an_unknown_archive_type(self, tmp_path):
        odd = tmp_path / "archive.rar"
        odd.write_bytes(b"not a supported archive")

        with pytest.raises(ValueError, match="Don't know how to extract"):
            extract(odd, tmp_path / "out")

    def test_refuses_a_tar_entry_escaping_the_destination(self, tmp_path):
        """filter='data' is what stops a malicious tarball writing outside dest."""
        evil = tmp_path / "evil.tar.gz"
        victim = tmp_path / "payload.txt"
        victim.write_text("original")
        with tarfile.open(evil, "w:gz") as tar:
            tar.add(victim, arcname="../escaped.txt")

        with pytest.raises(Exception):  # noqa: B017 - tarfile raises its own family here
            extract(evil, tmp_path / "out")


class TestDownloadAndExtract:
    def test_fetches_then_unpacks(self, tmp_path, tarball, fake_download):
        fake_download(tarball)
        dest = tmp_path / "data"

        download_and_extract("https://example.invalid/archive.tar.gz", dest)

        assert (dest / "payload" / "marker.txt").read_text() == "hello"

    def test_skips_everything_when_the_archive_is_present(self, tmp_path, monkeypatch):
        archive = tmp_path / "archive.tar.gz"
        archive.write_bytes(b"already here")

        def explode(*args, **kwargs):
            raise AssertionError("must not download when the archive exists")

        monkeypatch.setattr(archives.urllib.request, "urlretrieve", explode)

        assert download_and_extract("https://example.invalid/archive.tar.gz", tmp_path) == archive


class TestPrintDownloadProgress:
    def test_reports_a_percentage(self, capsys):
        print_download_progress(count=1, block_size=50, total_size=200)

        assert "25.0%" in capsys.readouterr().out

    def test_survives_an_unknown_total_size(self, capsys):
        """Servers that send no Content-Length report total_size as -1."""
        print_download_progress(count=1, block_size=50, total_size=-1)

        assert capsys.readouterr().out == ""
