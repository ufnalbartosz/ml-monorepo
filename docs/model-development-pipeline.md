# Building a modern model development pipeline

What a 2026 vision project looks like end to end, and which of it this repo
already has.

The short version: the modelling code is the small part. What makes a pipeline
work is that data, experiments, evaluation and deployment are each versioned,
reproducible and automated — and that you can answer "why is the model in
production behaving like this?" six months later.

---

## The eight stages

### 1. Data: versioned, not just present

The thing that most often breaks a model is a silent change in its input. So
the data-set needs a version like the code does.

- **Storage**: object store (S3/GCS) with content-addressed paths. Never "the
  latest export".
- **Versioning**: DVC, LakeFS, or a dataset table in Delta/Iceberg. The
  requirement is that a training run records *which* data-set hash it used.
- **Labelling**: Label Studio or CVAT if you own it; a vendor if you don't.
  Budget for a second pass — inter-annotator agreement below ~0.8 means your
  ceiling is the label noise, not the model.
- **Validation**: schema and distribution checks that run before training, not
  after it fails. Great Expectations, or a hand-rolled assertion module — the
  tool matters less than it being a gate.

*This repo:* `vision_core.cache` gives you the "build once, reload instantly"
half. There is no versioning — the pickle has no hash and no provenance. That's
the first thing you would add.

### 2. Splits: decided once, by a rule, and never by hand

The split logic must be deterministic and reproducible from the raw data. It
must also be *grouped correctly* — this is where real projects silently fail.
If you have multiple photos of the same person, product or physical part, all
of them go on the same side of the split, or your test accuracy is measuring
memorisation.

*This repo:* `pure_alexnet.dataset.train_test_valid_split` is a pure function
of the label matrix and is tested for disjointness and completeness, which is
exactly the right shape. It has no grouping concept, because the flowers
data-set has no groups.

### 3. Configuration: one object, serialised with the run

Every hyper-parameter in one typed config (Hydra, Pydantic Settings, or a plain
frozen dataclass), serialised into the run artifacts. If a number is in the
code and not in the config, you will not be able to reproduce the run.

*This repo:* `Cifar100Config` is this pattern. The training hyper-parameters
are still argparse flags, which is fine at this scale — the upgrade is one
config object covering data + model + optimizer, dumped next to the checkpoint.

### 4. Training: a loop you did not write

Use `keras.Model.fit`, PyTorch Lightning, or HF `Trainer`. Hand-rolled loops
are where mixed precision, gradient accumulation, distributed training and
checkpoint-resume bugs live.

What belongs in your code is the model, the data pipeline, and the callbacks.

*This repo:* `vision_core.training.train` is exactly this — a thin wrapper over
`fit` that supplies the callbacks and refuses to touch the test split.

### 5. Experiment tracking: automatic, not a spreadsheet

MLflow, Weights & Biases, Neptune, or Aim. Every run logs: git SHA, data
version, full config, metrics per epoch, and the artifact path. The test is
whether you can sort every run you have ever done by validation accuracy and
click through to the exact code and data of the best one.

TensorBoard — which both packages already write — is the metrics half of this
and none of the provenance half.

### 6. Evaluation: a report, not a number

A single accuracy figure hides everything that matters.

- **Per-class metrics**, always. A 92% average over 20 classes can be 99% on
  nineteen and 8% on one.
- **Confusion matrix**, and *look* at the confusions — they usually tell you
  the label taxonomy is wrong.
- **Slices**: accuracy by lighting, device, demographic, source. This is where
  fairness problems and dataset shortcuts show up, and it is where an aggregate
  number is actively misleading.
- **Calibration** (ECE, reliability diagrams) if a downstream system consumes
  the confidence. Modern networks are badly overconfident by default.
- **Failure gallery**: the worst N misclassifications, as images, every run.
- **Regression set**: a frozen set of cases that must never break. This is your
  unit-test suite for model behaviour.

*This repo:* `evaluate.py` plus `plot_example_errors` and
`plot_confusion_matrix` cover the first, second and fifth. Slices and
calibration are missing, and they are the two that matter most in production.

### 7. Packaging and serving

- **Export**: ONNX, TorchScript, or SavedModel — a format that doesn't need
  your training code to load.
- **Optimise**: quantise (INT8 is usually free accuracy-wise; INT4 needs
  checking), distil, or prune. Measure latency at your real batch size, on your
  real hardware, at p95 — not mean throughput on a benchmark.
- **Serve**: Triton, TorchServe, BentoML, or a plain FastAPI process if the
  load is small. Batch requests if latency budget allows.
- **Contract**: the preprocessing at serving time must be bit-identical to
  training. This is the single most common production bug in vision — a
  different resize interpolation or channel order between the two silently
  costs several points of accuracy.

### 8. Monitoring: the model degrades even when nothing changes

- **Input drift**: distribution distance on embeddings or summary statistics.
- **Prediction drift**: class balance of outputs over time.
- **Delayed ground truth**: wire up the eventual labels and compute real
  accuracy on a lag.
- **Feedback loop**: route low-confidence and drifted samples back to
  labelling. This is what makes the system improve rather than decay.

---

## The CI layer that ties it together

Three tiers, running at different frequencies:

**On every commit (seconds to a minute).** What this repo's suites already do:
build every model at reduced size and check the output shape; assert gradients
reach every trainable weight; check the data transformations against synthetic
arrays; run the CLI end to end on random data. No GPU, no network, no
downloads. This catches the overwhelming majority of "I broke the pipeline"
errors before a training run burns an hour.

**On every merge to main (minutes).** Train to a deliberately low bar on a
small subset and assert a floor — "reaches 40% on 5% of the data in 3 epochs".
This catches regressions that shape tests cannot: a broken normalisation, a
label misalignment, an LR that silently became zero.

**Nightly or on release (hours).** Full training run, full evaluation report,
comparison against the current production model on the regression set.

The pattern to internalise: **make the fast tier fast enough that nobody skips
it.** That is why both packages' models take `input_shape` and `dense_units` as
arguments — so a structurally identical network can be built at 32×32 with
8-unit dense layers and trained inside a unit test. That is not a testing trick;
it is the design property that makes a fast CI tier possible at all.

---

## What this repo would need next

Roughly in order of value per hour spent:

1. **Data versioning.** Hash the prepared pickle, record the hash in the run.
2. **Experiment tracking.** MLflow around `main()`, logging config + git SHA.
3. **Per-class and sliced metrics** in `evaluate.py`, not just the aggregate.
4. **A merge-gate training test** — 3 epochs on a subset, assert an accuracy
   floor.
5. **An export path** — SavedModel or ONNX, plus a test that the exported model
   and the Keras model agree on the same inputs to within tolerance.
6. **A serving preprocessing contract** — one function used by both training
   and inference, with a test that they produce identical output.

Items 3 and 6 are the ones that prevent silent, expensive production failures.
Items 1 and 2 are the ones you will regret not having the first time someone
asks "what changed?".
