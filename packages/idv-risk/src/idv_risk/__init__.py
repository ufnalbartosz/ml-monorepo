"""XGBoost risk-fusion engine for identity verification decisions.

Fuses the outputs of a verification stack - document checks, NFC chip
authentication, face matching, presentation attack detection, injection
detection, device attestation, velocity and graph features, behavioural
signals - into one calibrated fraud probability and an accept / step-up /
reject decision.

The design position, and the reason this package is the risk layer rather than
another biometric model: presentation attack detection and injection detection
are a permanent adversarial arms race best bought from a certified vendor, and
face recognition is close to solved for cooperative capture. What no vendor can
supply is the orchestration layer, because it encodes your risk appetite and
your cost structure.

Reach for the module you need:

* :mod:`idv_risk.schema` - the signal contract and its monotone constraints
* :mod:`idv_risk.synthetic` - the reference data-set, with four fraud typologies
* :mod:`idv_risk.features` - the one train/serve feature builder
* :mod:`idv_risk.splits` - group- and time-aware splitting
* :mod:`idv_risk.model` - the XGBoost classifier
* :mod:`idv_risk.calibration` - scores to probabilities
* :mod:`idv_risk.decision` - cost-optimal accept/step-up/reject bands
* :mod:`idv_risk.metrics` - PR-AUC, ISO 30107-3 and ISO 19795-10 reporting
* :mod:`idv_risk.artifact` - the deployable bundle
* :mod:`idv_risk.pipeline` - the whole thing wired together
* :mod:`idv_risk.cli` - train / evaluate / score
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
