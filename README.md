# ml-monorepo

Four machine-learning and numerical-optimization projects in one uv workspace,
each independently installable, sharing a single dependency lockfile.

| package | what it is | status |
|---|---|---|
| `packages/nadaraya-watson` | Nadaraya-Watson kernel regression demo | working |
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

The two CNN suites run on CPU in well under a minute and never touch the
network: data loaders are injected, and the models take `input_shape` and
layer-width arguments so a structurally identical network can be built small
enough to train inside a test.

`packages/dfp-formula-project/tests/test_gui.py` needs PyQt6's X client
libraries and fails on a bare headless machine.
