# ml-monorepo

Four consolidated projects, each independently installable, sharing one
dependency lockfile.

| package | what it is | status |
|---|---|---|
| `packages/nadaraya-watson` | Nadaraya-Watson kernel regression demo | working |
| `packages/dfp-formula-project` | DFP quasi-Newton optimizer, Qt GUI, 3D plots | working |
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

## Tests

```bash
uv run pytest
```

## History

Each package kept its original commit history:

```bash
git log packages/project-cnn/
```

The untouched pre-import histories, with their original commit SHAs, are on the
`archive/*` branches. They are reference-only and never merged.
