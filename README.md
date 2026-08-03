# ml-monorepo

Five machine-learning and numerical-optimization packages in one uv workspace,
each independently installable, sharing a single dependency lockfile.

| package | what it is | status |
|---|---|---|
| `packages/nadaraya-watson` | Nadaraya-Watson kernel regression demo | working |
| `packages/vision-core` | shared data, caching, training and metrics helpers | working |
| `packages/dfp-formula-project` | DFP quasi-Newton optimizer, Qt GUI, contour plots | working |
| `packages/pure-alexnet` | AlexNet on the Oxford 17-flowers set, Keras 3 | working |
| `packages/project-cnn` | Three CNNs on a CIFAR-100 subset, Keras 3 | working |

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
```

`nadaraya_watson.nw_regression` writes its plot to the working directory. The
two CNN packages download their data-sets on first run and cache the prepared
splits next to them; see each package's README.

## Tests

```bash
uv run pytest                       # everything
uv run pytest packages/project-cnn  # one package
```

The three vision suites run on CPU in about a minute and never touch the
network: data loaders are injected, and the models take `input_shape` and
layer-width arguments so a structurally identical network can be built small
enough to train inside a test.

`pure-alexnet` and `project-cnn` are separate packages because they train
different models on different data-sets; what they genuinely shared — archive
download, the pickle cache, the Keras training loop, accuracy — lives in
`vision-core` and is tested once.

`packages/dfp-formula-project/tests/test_gui.py` needs PyQt6's X client
libraries and fails on a bare headless machine.

## Further reading

- [`docs/modern-vision-guide.md`](docs/modern-vision-guide.md) — how the models
  in this repo relate to 2026 computer vision, a four-step path for closing the
  gap using this repo's own data-set, and where classical image processing is
  still the right tool.
- [`docs/model-development-pipeline.md`](docs/model-development-pipeline.md) —
  what a production ML pipeline looks like end to end, which stages this repo
  already has, and what to add next.
- [`docs/identity-verification-models.md`](docs/identity-verification-models.md)
  — state of the art in identity verification and fraud detection: matchers,
  presentation and injection attack detection, evaluation, and the regulatory
  constraints that shape the architecture.
