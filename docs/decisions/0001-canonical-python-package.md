# 0001: Canonical Python Package (`dggr/`)

Date: 2026-03-02

## Context

The repo historically contained two separate Python packages both named `src/`:
- `lab 2/src` for Lab 2 code
- `lab 3/src` for Lab 3/4 code

This made imports and documentation confusing for collaborators and increased the likelihood of stale/duplicated code.

## Decision

Maintain a single canonical top-level Python package:
- `dggr/` contains the authoritative code for Labs 2–4.
- `lab 2/src` and `lab 3/src` remain as compatibility shims (`from dggr.<module> import *`) so existing run scripts continue to work.

## Consequences

Positive:
- One place to read/modify core logic.
- Easier import paths for docs and notebooks.

Negative:
- Requires minimal coordination if someone edits legacy `lab */src` wrappers (they should not).

