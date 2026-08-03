"""Shared fixtures.

The expensive fixtures are session-scoped: training the model once and reusing
it across every test that needs a fitted estimator keeps the whole suite under
a minute, which is the difference between a check that runs on every commit and
one that gets skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from idv_risk.config import ModelConfig, TrainingConfig
from idv_risk.pipeline import train
from idv_risk.synthetic import GeneratorConfig, generate

#: Big enough for four typologies to be present in every split and for
#: isotonic calibration to have something to fit; small enough to train in
#: a couple of seconds.
SUITE_ROWS = 6_000


@pytest.fixture(scope="session")
def frame():
    """A generated application table, shared across the suite."""
    return generate(GeneratorConfig(n_applications=SUITE_ROWS, seed=7))


@pytest.fixture
def small_frame():
    """A smaller table for tests that do not need statistical power."""
    return generate(GeneratorConfig(n_applications=800, seed=11))


@pytest.fixture(scope="session")
def fast_config() -> TrainingConfig:
    """Training config trimmed for test speed, structurally unchanged."""
    return TrainingConfig(
        model=ModelConfig(n_estimators=60, early_stopping_rounds=10, max_depth=4, n_jobs=2)
    )


@pytest.fixture(scope="session")
def trained(frame, fast_config):
    """A fitted :class:`~idv_risk.pipeline.TrainingResult`."""
    return train(frame, fast_config)


@pytest.fixture(scope="session")
def model(trained):
    return trained.model


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)
