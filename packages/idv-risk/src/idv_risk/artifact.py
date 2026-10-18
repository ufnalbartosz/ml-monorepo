"""The deployable bundle: model, calibrator, thresholds, contract, provenance.

A saved model on its own is not deployable. Scoring an application needs the
booster *and* the calibrator *and* the two thresholds *and* the feature order —
and if any of the four drifts from the others, the system keeps returning
numbers, just wrong ones. So they are saved and loaded as one unit, and
:meth:`RiskModel.score` is the only supported way to get a decision.

Provenance travels with it: config, git SHA, a hash of the training data, the
library versions and the metrics at training time. "Why is the model rejecting
3% of applicants today?" is asked under pressure, and the answer starts with
knowing exactly which model is deployed.

The booster uses XGBoost's native JSON rather than pickle. Pickle ties the
artifact to the exact Python and library versions that wrote it, and a
five-month-old pickle that no longer loads is a bad day.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from idv_risk.config import TrainingConfig
from idv_risk.decision import Thresholds
from idv_risk.features import build_features, validate_features

BOOSTER_FILE = "booster.json"
CALIBRATOR_FILE = "calibrator.joblib"
METADATA_FILE = "metadata.json"

ARTIFACT_VERSION = 1


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Stable content hash of a frame, for recording which data trained a model.

    Uses pandas' row hashing over the sorted column set, so column order does
    not change the fingerprint but content does. Enough to answer "was this
    model trained on the data I think it was?", which is the whole job.
    """
    ordered = frame[sorted(frame.columns)]
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(row_hashes))
    digest.update(",".join(sorted(frame.columns)).encode())

    return digest.hexdigest()


def git_revision() -> str | None:
    """Current commit SHA, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout.strip() or None if result.returncode == 0 else None


@dataclass(frozen=True)
class Provenance:
    """Everything needed to explain where a model came from."""

    created_at: str
    git_revision: str | None
    training_rows: int
    training_fraud_rate: float
    data_fingerprint: str
    library_versions: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        label_column: str,
        metrics: dict[str, Any] | None = None,
    ) -> Provenance:
        return cls(
            created_at=datetime.now(timezone.utc).isoformat(),
            git_revision=git_revision(),
            training_rows=len(frame),
            training_fraud_rate=float(frame[label_column].mean()),
            data_fingerprint=dataframe_fingerprint(frame),
            library_versions={
                "xgboost": xgb.__version__,
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
            metrics=metrics or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "git_revision": self.git_revision,
            "training_rows": self.training_rows,
            "training_fraud_rate": self.training_fraud_rate,
            "data_fingerprint": self.data_fingerprint,
            "library_versions": self.library_versions,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Provenance:
        return cls(
            created_at=payload["created_at"],
            git_revision=payload.get("git_revision"),
            training_rows=payload["training_rows"],
            training_fraud_rate=payload["training_fraud_rate"],
            data_fingerprint=payload["data_fingerprint"],
            library_versions=payload.get("library_versions", {}),
            metrics=payload.get("metrics", {}),
        )


@dataclass
class RiskModel:
    """A scoring engine that can be saved, loaded and served.

    :param classifier: the fitted XGBoost model.
    :param calibrator: wraps the classifier and maps its scores to
        probabilities. Scoring always goes through this, never the raw
        classifier, because the thresholds are calibrated-probability bands.
    """

    classifier: xgb.XGBClassifier
    calibrator: Any
    thresholds: Thresholds
    feature_names: tuple[str, ...]
    config: TrainingConfig
    provenance: Provenance

    def risk_score(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated P(fraud) for raw applications.

        Takes the *raw* frame, not built features, so a caller cannot
        accidentally use a different feature-building path from the one
        training used. That is the train/serve contract, enforced by API shape
        rather than by documentation.
        """
        features = build_features(frame, self.feature_names)
        validate_features(features, self.feature_names)

        return self.calibrator.predict_proba(features)[:, 1]

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        """Accept / step-up / reject per application."""
        return self.thresholds.decide(self.risk_score(frame))

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scores and decisions together — the serving entrypoint."""
        probabilities = self.risk_score(frame)

        return pd.DataFrame(
            {
                "risk_score": probabilities,
                "decision": self.thresholds.decide(probabilities),
            },
            index=frame.index,
        )

    def save(self, directory: Path | str) -> Path:
        """Write the bundle to ``directory``, creating it if needed."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.classifier.save_model(directory / BOOSTER_FILE)
        joblib.dump(self.calibrator, directory / CALIBRATOR_FILE)

        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "feature_names": list(self.feature_names),
            "thresholds": self.thresholds.to_dict(),
            "config": self.config.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
        (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True))

        return directory

    @classmethod
    def load(cls, directory: Path | str) -> RiskModel:
        """Read a bundle back. Scores identically to the model that wrote it."""
        directory = Path(directory)

        metadata_path = directory / METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"no {METADATA_FILE} in {directory}")

        metadata = json.loads(metadata_path.read_text())

        version = metadata.get("artifact_version")
        if version != ARTIFACT_VERSION:
            raise ValueError(
                f"artifact version {version} is not supported by this build "
                f"(expected {ARTIFACT_VERSION})"
            )

        classifier = xgb.XGBClassifier()
        classifier.load_model(directory / BOOSTER_FILE)

        return cls(
            classifier=classifier,
            calibrator=joblib.load(directory / CALIBRATOR_FILE),
            thresholds=Thresholds.from_dict(metadata["thresholds"]),
            feature_names=tuple(metadata["feature_names"]),
            config=TrainingConfig.from_dict(metadata["config"]),
            provenance=Provenance.from_dict(metadata["provenance"]),
        )
