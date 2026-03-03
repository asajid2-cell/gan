# Contributing

This repo is primarily a research codebase. The goal of these guidelines is to
keep experiments reproducible and collaboration friction low.

## Project conventions

- Canonical Python package: `dggr/`
- Legacy lab entrypoints: `lab 2/` and `lab 3/` (kept runnable)
- Notebooks live in `notebooks/` and are committed **without outputs**
- Documentation lives in `docs/` (Diataxis structure)

## Before opening a PR

- Keep notebooks clean:
  - Clear outputs before committing (to keep diffs small and readable).
- Keep path defaults portable:
  - Prefer repo-relative paths (e.g. `saves2/...`) rather than machine-specific `Z:\...`
- Record results:
  - Add run IDs, key metrics, and any qualitative notes to the relevant doc page.

## Writing docs

- Tutorials: a clean walkthrough with copy-paste commands.
- How-to: one goal, multiple options, minimal explanation.
- Reference: exact flags, shapes, file formats.
- Explanation: rationale, tradeoffs, design constraints.

See `docs/README.md` and `docs/references.md`.
