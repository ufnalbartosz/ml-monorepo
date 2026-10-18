"""Train/validation/test splits that do not leak.

**Time is the primary axis.** Fraud is adversarial and non-stationary: last
quarter's typology mix is not this quarter's. A model evaluated on a random
sample of history is being asked an easier question than production asks, which
is always "generalise forward". So the periods are cut in time order — train on
the past, validate on the middle, test on the future.

**Then applicants are purged.** The same person on both sides of a boundary is
straightforward label leakage; their outcome is the same event. Any row in a
later period whose applicant was already seen in an earlier one is dropped.

**Devices are deliberately not purged, and this is the subtle part.** The
tempting move is to purge every shared entity — applicant, device, IP — so that
nothing whatsoever connects the periods. Two things go wrong. Statistically,
shared devices form a giant connected component: on a realistic table almost
every row ends up transitively linked to almost every other, and there is no
split left to make. More importantly it would be measuring the wrong thing.
Sharing a device with a previously-flagged applicant is not leakage, it is the
single most valuable signal a fraud graph produces, and in production that
history *is* available at scoring time. Purging it would evaluate the model on
a world where ``graph_ring_size`` cannot mean anything — and since detecting
organised rings is what the graph stage exists for, that evaluation would be
worse than useless.

So: purge identity, keep the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Entities whose reappearance across a boundary is genuine leakage. Devices
#: and IPs are absent on purpose; see the module docstring.
DEFAULT_PURGE_COLUMNS: tuple[str, ...] = ("applicant_id",)


@dataclass(frozen=True)
class Split:
    """Row index labels for the three periods."""

    train: pd.Index
    valid: pd.Index
    test: pd.Index

    def __iter__(self):
        return iter((self.train, self.valid, self.test))

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.valid), len(self.test)

    def frames(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return frame.loc[self.train], frame.loc[self.valid], frame.loc[self.test]


def time_split(
    frame: pd.DataFrame,
    time_column: str = "timestamp",
    valid_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Split:
    """Cut the frame into three consecutive periods by time.

    No purging yet — :func:`purge_seen_groups` does that, and keeping the two
    apart means each can be tested for exactly one property.
    """
    _validate_fractions(valid_fraction, test_fraction)
    if time_column not in frame.columns:
        raise KeyError(f"frame has no column {time_column!r}")

    ordered = frame[time_column].sort_values()
    n = len(ordered)
    if n < 3:
        raise ValueError(f"need at least 3 rows to split, got {n}")

    n_test = int(round(n * test_fraction))
    n_valid = int(round(n * valid_fraction))
    n_train = n - n_valid - n_test

    if n_train < 1:
        raise ValueError(
            f"fractions leave no training rows: {n} rows, "
            f"valid_fraction={valid_fraction}, test_fraction={test_fraction}"
        )

    index = ordered.index

    return Split(
        train=index[:n_train],
        valid=index[n_train : n_train + n_valid],
        test=index[n_train + n_valid :],
    )


def purge_seen_groups(
    frame: pd.DataFrame,
    split: Split,
    group_columns: tuple[str, ...] = DEFAULT_PURGE_COLUMNS,
) -> Split:
    """Drop later-period rows whose group already appeared in an earlier one.

    Rows are dropped rather than moved backwards: moving them would violate the
    time ordering that the split exists to establish. The cost is a slightly
    smaller evaluation set, which is the cheaper of the two errors.
    """
    for column in group_columns:
        if column not in frame.columns:
            raise KeyError(f"frame has no column {column!r}")

    if not group_columns:
        return split

    def seen_values(index: pd.Index) -> dict[str, set]:
        return {column: set(frame.loc[index, column].dropna()) for column in group_columns}

    seen = seen_values(split.train)

    def keep(index: pd.Index) -> pd.Index:
        mask = pd.Series(True, index=index)
        for column in group_columns:
            mask &= ~frame.loc[index, column].isin(seen[column])
        return index[mask.to_numpy()]

    valid = keep(split.valid)
    for column in group_columns:
        seen[column] |= set(frame.loc[valid, column].dropna())

    test = keep(split.test)

    return Split(train=split.train, valid=valid, test=test)


def time_grouped_split(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...] = DEFAULT_PURGE_COLUMNS,
    time_column: str = "timestamp",
    valid_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Split:
    """A forward-in-time split with repeat applicants purged from later periods."""
    split = time_split(frame, time_column, valid_fraction, test_fraction)
    return purge_seen_groups(frame, split, group_columns)


def _validate_fractions(valid_fraction: float, test_fraction: float) -> None:
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("valid_fraction + test_fraction must be below 1.0")


def assert_no_group_overlap(
    frame: pd.DataFrame,
    split: Split,
    group_columns: tuple[str, ...] = DEFAULT_PURGE_COLUMNS,
) -> None:
    """Raise if any purged group value appears in more than one period.

    Worth calling in the pipeline, not only in tests: a change that reintroduces
    leakage inflates every downstream number, and inflated numbers do not look
    like a bug.
    """
    for column in group_columns:
        seen = {
            name: set(frame.loc[index, column].dropna())
            for name, index in (
                ("train", split.train),
                ("valid", split.valid),
                ("test", split.test),
            )
        }
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = seen[left] & seen[right]
            if overlap:
                raise AssertionError(
                    f"{column}: {len(overlap)} value(s) shared between {left} and {right}, "
                    f"e.g. {sorted(overlap)[:3]}"
                )


def assert_chronological(
    frame: pd.DataFrame,
    split: Split,
    time_column: str = "timestamp",
) -> None:
    """Raise if any period overlaps a later one in time."""
    bounds = {
        name: (frame.loc[index, time_column].min(), frame.loc[index, time_column].max())
        for name, index in (("train", split.train), ("valid", split.valid), ("test", split.test))
        if len(index)
    }

    order = [name for name in ("train", "valid", "test") if name in bounds]
    for earlier, later in zip(order, order[1:], strict=False):
        if bounds[earlier][1] > bounds[later][0]:
            raise AssertionError(
                f"{earlier} extends past the start of {later}: "
                f"{bounds[earlier][1]} > {bounds[later][0]}"
            )
