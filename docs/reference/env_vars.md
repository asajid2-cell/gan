# Environment Variables

The codebase supports portable defaults via environment variables.

## `DGGR_DATA_ROOT`

Purpose:
- Root directory where datasets live (audio corpora, rendered symbolic audio, soundfonts).

Default behavior:
1. If `DGGR_DATA_ROOT` is set: use it.
2. Else, if `Z:/DataSets` exists: use it.
3. Else: fall back to `./data`.

## `DGGR_MANIFESTS_ROOT`

Purpose:
- Directory containing cleaned manifest CSVs (the inputs to Lab 2 / Lab 3 sampling).

Default behavior:
1. If `DGGR_MANIFESTS_ROOT` is set: use it.
2. Else, if `Z:/DataSets/_lab1_manifests` exists: use it.
3. Else: fall back to `./data/_lab1_manifests`.

## `.env` workflow (recommended)

Copy `.env.example` to `.env` and set the paths that match your machine.
This repo does not auto-load `.env`, but many IDEs and shells will.

