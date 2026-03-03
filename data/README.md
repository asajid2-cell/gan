# Placeholder data folder

This repo expects large datasets and manifests that are not committed to git.

Defaults:
- If `DGGR_DATA_ROOT` exists, use it.
- Else if `Z:/DataSets` exists, use that.
- Else fall back to `./data`.

Manifests:
- Set `DGGR_MANIFESTS_ROOT` to point at your cleaned manifest directory.
- Default fallback is `<DATA_ROOT>/_lab1_manifests`.
