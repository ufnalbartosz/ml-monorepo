# idv-risk

An XGBoost risk-fusion engine for identity verification decisions. It takes the
outputs of a verification stack — document checks, NFC chip authentication,
face matching, presentation attack detection, injection detection, device
attestation, velocity and graph features, behavioural signals — and produces
one calibrated fraud probability plus an **accept / step-up / reject** decision.

## Why this layer, and not another biometric model

Face recognition is close to solved for cooperative capture: in NIST's FRTE 1:1
evaluation the leading algorithms reach a false-non-match rate well under 1% at
a false-match rate of 1 in a million. Presentation attack detection and
injection detection are a permanent adversarial arms race, best bought from a
vendor carrying iBeta/ISO 30107-3 certification and the red-team capability to
keep it.

What no vendor can supply is the orchestration layer, because it encodes *your*
risk appetite and *your* cost structure. That is this package.

The threat landscape backs the emphasis up: through 2025–2026 deepfake use in
biometric fraud attempts rose sharply and **injection** attacks — feeding frames
straight into the stream via a virtual camera or SDK hook, bypassing the camera
and therefore PAD entirely — grew faster than presentation attacks. Gartner's
much-quoted line, that by 2026 a large share of enterprises would stop trusting
standalone identity verification in isolation, is the same conclusion from the
buyer's side.

## The core claim, and how it is tested

**No single signal family catches everything.** The synthetic data-set encodes
this by giving each fraud typology a different blind spot:

| typology | what it defeats |
|---|---|
| `presentation` | Nothing clever — PAD sees it. The easy case. |
| `injection` | PAD sees a *perfect* face: high match, no artefacts, because the frames never met a camera. Only attestation and frame timing fire. |
| `document_forgery` | Face and liveness are a real person; the document is not. Only the document stage and the absent NFC chip fire. |
| `synthetic_id` | Every per-session signal is clean, because every session *is* clean. Only velocity and graph features fire. |

`TestAblations` turns that table into assertions: a model trained on biometric
and PAD signals alone catches presentation attacks and is **blind to the
synthetic-identity rings**; a graph-only model is the mirror image; the fused
model beats both. If those assertions ever stop holding, this package has lost
its reason to exist.

## Design decisions worth knowing about

**Monotone constraints, but sparsely.** XGBoost can be told its output must not
decrease in a given feature. Where the direction is unambiguous — tamper score,
virtual camera detected, NFC passive authentication — constraining it buys
robustness against distribution shift and an explanation that survives an
audit. Two signals that *look* obviously monotone are deliberately left free:

- `face_match_score`, because an injection or deepfake attack **optimises it
  upward**. Constraining risk to fall as match rises would hand an attacker a
  guaranteed direction to push.
- `doc_ocr_confidence`, because a clean forgery reads better than a worn
  genuine passport.

18 of 28 signals are constrained. `test_monotone_constraint_holds_end_to_end`
probes the fitted booster by sweeping each one, so a constraint that was
declared but not applied fails the build.

**Missing values are not imputed.** NFC is absent for most sessions because of
the handset, not the applicant. XGBoost learns a default branch direction per
split, so "no chip was read" is a fact the model uses. This is a substantive
reason to prefer a tree ensemble here over anything needing imputation.

**Categorical levels are pinned.** XGBoost *raises* on a categorical level it
did not see in training — verified against the installed version before the
package was designed around it. A new device platform would therefore be an
outage, not a degradation, so the feature builder folds unknown levels into a
reserved `__unknown__` category.

**Calibration is a dependency, not a nicety.** The decision bands are absolute
probability thresholds, so the arithmetic multiplies a probability by a cost.
Boosted ensembles trained with `scale_pos_weight` are doubly miscalibrated.
Isotonic regression on the validation split, falling back to sigmoid when
there are too few positives for a non-parametric fit. The wrapper is
`CalibratedClassifierCV(FrozenEstimator(clf))` — `cv="prefit"` was deprecated
in scikit-learn 1.6.

**Thresholds come from costs, not accuracy.** At a 4% fraud rate the
accuracy-maximising policy is to accept everything. `optimise_thresholds`
grid-searches the boundary pair minimising expected cost, with `step_up` priced
honestly: a review costs analyst time, misses ~10% of fraud anyway, and ~15% of
genuine applicants abandon rather than complete one. Without those terms the
optimiser routes everything to a review team that does not exist. There is also
a hard cap on the step-up rate, because a cost-optimal policy that reviews 40%
of traffic is undeployable.

**The split is temporal, and purges applicants but not devices.** Fraud is
non-stationary, so train on the past and test on the future. The same applicant
on both sides is leakage. Devices are *deliberately* not purged: sharing a
device with a flagged applicant is the single most valuable graph signal, and
in production that history is available at scoring time. Purging it also
collapses the split — shared devices form a giant connected component — and
during development it silently deleted the entire `synthetic_id` typology from
the test set. `test_every_typology_survives_into_the_test_period` guards that
regression.

## Layout

| module | responsibility |
|---|---|
| `schema.py` | the signal contract and its monotone directions |
| `synthetic.py` | the reference data-set, four fraud typologies |
| `features.py` | the one train/serve feature builder |
| `splits.py` | temporal split with applicant purging |
| `model.py` | the XGBoost classifier |
| `calibration.py` | scores to probabilities, plus ECE and reliability |
| `decision.py` | cost-optimal accept/step-up/reject bands |
| `metrics.py` | PR-AUC, recall@FPR, ISO 30107-3, ISO 19795-10 |
| `artifact.py` | the deployable bundle and its provenance |
| `pipeline.py` | the whole thing wired together |
| `cli.py` | train / evaluate / score |

## Running

```bash
uv run python -m idv_risk.cli train --rows 20000 --out models/candidate
uv run python -m idv_risk.cli evaluate --model models/candidate
uv run python -m idv_risk.cli score --model models/candidate --input apps.parquet
```

Every subcommand falls back to the synthetic generator, so it runs end to end
on a fresh checkout with nothing downloaded. On the default 20 000-row
population expect a test PR-AUC around 0.93 and an ECE under 0.01.

To use real data, pass `--data` a parquet/CSV table carrying the columns in
`schema.SIGNALS` plus `applicant_id`, `device_id`, `timestamp` and `is_fraud`.

## Reporting

`metrics.py` deliberately does not report accuracy. It reports:

- **PR-AUC** as the headline — ROC-AUC barely moves under this imbalance.
- **Recall at fixed FPR**, because the FPR is the friction budget and the
  business fixes it, not the model.
- **APCER / BPCER** at a stated threshold, in ISO/IEC 30107-3's vocabulary, so
  the fusion decision can be discussed with an assessor who thinks in those
  terms. Always as a pair — either alone is meaningless.
- **Per-typology recall**, where fusion justifies itself.
- **Subgroup differentials** with the ISO/IEC 19795-10:2024 Inequity Rate.
  Error rates in biometric systems vary across demographic groups; an aggregate
  number conceals exactly the failure that produces both harm and litigation.
  Rates are Jeffreys-smoothed and groups need enough of *each* class to count,
  so the metric reports unfairness rather than small-sample noise.

## Data protection

The generator exists because real IDV data is biometric data: special-category
under GDPR Article 9, and in Illinois covered by BIPA with a private right of
action and a history of nine-figure settlements. It cannot go in a repository,
and a package whose tests need it is a package nobody can run.

If you point this at real data: store templates rather than images, set a
retention schedule you can enforce, and measure the subgroup differentials
above before deployment rather than after a complaint.

## Tests

```bash
uv run pytest packages/idv-risk
```

236 tests, ~13 seconds, no network access and no downloads.
