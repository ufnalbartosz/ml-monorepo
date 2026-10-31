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
