# ml-monorepo

Four machine-learning and numerical-optimization projects in one uv workspace,
each independently installable, sharing a single dependency lockfile.

| package | what it is | status |
|---|---|---|
| `packages/nadaraya-watson` | Nadaraya-Watson kernel regression demo | working |
| `packages/dfp-formula-project` | DFP quasi-Newton optimizer, Qt GUI, contour plots | working |
| `packages/pure-alexnet` | AlexNet on tflearn | installs; port pending |
| `packages/project-cnn` | Inception-style CNN for CIFAR-100 | installs; TF1 port pending |

## Setup

```bash
uv sync
```

## Running

```bash
uv run python -m nadaraya_watson.nw_regression
uv run python -m dfp_formula_project.main
```

`nadaraya_watson.nw_regression` writes its plot to the working directory.

## Tests

```bash
uv run pytest
```

Test coverage is currently limited to `packages/dfp-formula-project`.
