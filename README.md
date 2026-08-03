# ml-monorepo

Five machine-learning and numerical-optimization projects in one uv workspace,
each independently installable, sharing a single dependency lockfile.

| package | what it is | status |
|---|---|---|
| `packages/nadaraya-watson` | Nadaraya-Watson kernel regression demo | working |
| `packages/dfp-formula-project` | DFP quasi-Newton optimizer, Qt GUI, contour plots | working |
| `packages/pure-alexnet` | AlexNet on the Oxford 17-flowers set, Keras 3 | working |
| `packages/project-cnn` | Three CNNs on a CIFAR-100 subset, Keras 3 | working |
| `packages/idv-risk` | XGBoost risk-fusion engine for identity verification | working |

## Setup

```bash
uv sync
```

## Running

```bash
uv run python -m nadaraya_watson.nw_regression
uv run python -m dfp_formula_project.main
uv run python -m pure_alexnet.train --help
uv run python -m project_cnn.main --help
uv run python -m idv_risk.cli train --help
```

`nadaraya_watson.nw_regression` writes its plot to the working directory. The
two CNN packages download their data-sets on first run and cache the prepared
splits next to them; see each package's README.

## Tests

```bash
uv run pytest                       # everything
uv run pytest packages/project-cnn  # one package
```

The two CNN suites run on CPU in well under a minute and never touch the
network: data loaders are injected, and the models take `input_shape` and
layer-width arguments so a structurally identical network can be built small
enough to train inside a test.

`packages/idv-risk` needs no data at all: it generates its own synthetic
population, because real identity-verification data is biometric data and
cannot live in a repository.

`packages/dfp-formula-project/tests/test_gui.py` needs PyQt6's X client
libraries and fails on a bare headless machine.

## Further reading

- [`docs/modern-vision-guide.md`](docs/modern-vision-guide.md) — how the models
  in this repo relate to 2026 computer vision, a four-step path for closing the
  gap using this repo's own data-set, and where classical image processing is
  still the right tool.
- [`packages/idv-risk/README.md`](packages/idv-risk/README.md) — the design
  rationale behind the risk-fusion engine: why fusion rather than another
  biometric model, which monotone constraints are safe, and how the evaluation
  is built to report unfairness rather than noise.
