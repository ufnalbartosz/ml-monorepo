"""Downloading and unpacking data-set archives.

Both packages fetch a tarball over HTTPS and unpack it.  They had two copies of
this, one of which was still calling the Python 2 ``urllib.urlretrieve``.

``download`` is a separate function from ``extract`` on purpose: tests inject a
stand-in for the first and use the real second one against a fixture tarball.
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


def print_download_progress(count: int, block_size: int, total_size: int) -> None:
    """``urlretrieve`` reporthook that overwrites its own line."""
    if total_size <= 0:
        return

    pct_complete = float(count * block_size) / total_size
    sys.stdout.write(f"\r- Download progress: {pct_complete:.1%}")
    sys.stdout.flush()


def download(url: str, download_dir: Path | str) -> Path:
    """Fetch ``url`` into ``download_dir``, skipping it if the file is already there."""
    download_dir = Path(download_dir)
    archive_path = download_dir / url.rsplit("/", 1)[-1]

    if archive_path.exists():
        print(f"Archive already downloaded: {archive_path}")
        return archive_path

    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(  # noqa: S310 - callers pass a fixed https URL
        url=url,
        filename=archive_path,
        reporthook=print_download_progress,
    )
    print()
    print(f"Saved to {archive_path}")

    return archive_path


def extract(archive_path: Path | str, dest_dir: Path | str) -> Path:
    """Unpack a .zip or .tar.gz into ``dest_dir`` and return that directory."""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    name = archive_path.name
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            archive.extractall(dest_dir)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            # filter='data' refuses absolute paths and symlinks escaping dest_dir.
            archive.extractall(dest_dir, filter="data")
    else:
        raise ValueError(f"Don't know how to extract {archive_path}")

    return dest_dir


def download_and_extract(url: str, download_dir: Path | str) -> Path:
    """Download ``url`` and unpack it in place, skipping both if already done."""
    download_dir = Path(download_dir)
    archive_path = download_dir / url.rsplit("/", 1)[-1]

    if archive_path.exists():
        print("Data has apparently already been downloaded and unpacked.")
        return archive_path

    archive_path = download(url, download_dir)

    print("Download finished. Extracting files.")
    extract(archive_path, download_dir)
    print("Done.")

    return archive_path
