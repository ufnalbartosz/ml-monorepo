"""Pickle-backed caching for prepared data-sets.

Both CNN packages download something slow, turn it into a dict of numpy
arrays, and want that dict back instantly next time.  They had separate copies
of this; it lives here now.

Paths are always arguments - never ``os.getcwd()`` plus a hard-coded relative
name - which is what lets callers point the whole thing at ``tmp_path``.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def ensure_directories(*paths: Path | str) -> None:
    """Create the given directories if they do not exist yet."""
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            print(f"Creating {path} directory...")
            path.mkdir(parents=True, exist_ok=True)


def read_pickle(path: Path | str) -> object:
    """Load a pickle, with a clearer error than ``open`` gives for a missing file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File '{path}' does not exist.")

    with path.open("rb") as fp:
        return pickle.load(fp)


def write_pickle(obj: object, path: Path | str) -> Path:
    """Write ``obj`` to ``path``, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_or_build(path: Path | str, build: Callable[[], T]) -> T:
    """Return the cached object at ``path``, or build it and cache it.

    ``build`` is only called on a miss, which is what the tests assert against:
    a second call must not reach the network.
    """
    path = Path(path)

    if path.exists():
        print(f"Loading cached data from {path}")
        return read_pickle(path)  # type: ignore[return-value]

    print(f"No cache at {path}; building it")
    obj = build()
    write_pickle(obj, path)

    return obj
