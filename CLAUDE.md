# ml-monorepo

## Git workflow (user preference)

- Keep history linear, always. Update a branch with `git rebase origin/main`, never
  `git merge main`. No merge commits on feature branches. After rebasing your own
  branch, push with `git push --force-with-lease`.
- Work in git worktrees (`git worktree add ../<name> <branch>`), not by switching
  branches in the main checkout. Remove the worktree when done.
- Commit messages follow Conventional Commits: `type(scope): summary`, e.g.
  `feat(pure-alexnet): ...`, `fix: ...`, `docs: ...`, `test: ...`, `refactor: ...`,
  `chore: ...`.

## Workspace (uv)

- One uv workspace, one `uv.lock`, Python 3.12 pinned in `.python-version`.
  Don't change the pin without asking (TensorFlow wheels depend on it).
- Adding a package: add it to `[tool.uv.workspace].members`, `[tool.uv.sources]`
  and root `dependencies` in `pyproject.toml`, and to
  `[lint.isort].known-first-party` in `ruff.toml`. Then `uv lock`.
- Work on one package: `uv sync --package <name>`, `uv run --package <name> ...`.
- Never hand-edit `uv.lock`; regenerate it with `uv lock`.

## Lint and tests

- Before every commit: `uv run ruff format .`, `uv run ruff check .` and
  `uv run pytest packages/<changed-package>`. The whole repo is ruff-formatted;
  `uv run ruff format --check .` must pass.
- Use the ruff version pinned in `uv.lock` (`uv run ruff`), not a newer global
  one, so formatting doesn't drift between machines.
- Formatting churn in files you didn't otherwise change goes in its own `style:`
  commit, never mixed into a feature commit.
- Ruff config lives only in the root `ruff.toml`. No per-package configs; add
  per-file ignores only with a comment explaining why.
- Tests are offline: no network, no full-size models, no real datasets. Inject
  loaders and use synthetic data.
- `packages/dfp-formula-project/tests/test_gui.py` needs PyQt6's X libraries and
  fails on a headless machine; that is expected, not a regression.
