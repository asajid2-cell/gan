from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.validator import normalize


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


HEADER_MD = """# DGGR Notebooks

This notebook is part of the **Deep Generative Genre Remastering (DGGR)** project.

## How To Run
- Prefer running from the repo root so relative paths resolve.
- Most notebooks assume you have the Lab artifacts under `saves/` and `saves2/` (ignored by git).
- See `docs/` for setup, data layout, and reproduction notes.

## Notes
- Outputs are intentionally stripped for version control cleanliness.
"""


def strip_outputs(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        cell["outputs"] = []
        cell["execution_count"] = None
        # Clear noisy per-run UI state but keep structural metadata.
        cell.get("metadata", {}).pop("collapsed", None)
        cell.get("metadata", {}).pop("scrolled", None)


def ensure_header(nb: nbformat.NotebookNode) -> None:
    if not nb.cells:
        nb.cells.insert(0, nbformat.v4.new_markdown_cell(HEADER_MD))
        return
    first = nb.cells[0]
    if first.get("cell_type") == "markdown" and "DGGR Notebooks" in (first.get("source") or ""):
        return
    nb.cells.insert(0, nbformat.v4.new_markdown_cell(HEADER_MD))


def main() -> int:
    if not NOTEBOOKS_DIR.exists():
        raise SystemExit(f"Missing notebooks dir: {NOTEBOOKS_DIR}")

    paths = sorted(p for p in NOTEBOOKS_DIR.glob("*.ipynb") if p.is_file())
    if not paths:
        print("No notebooks found.")
        return 0

    touched = 0
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        _, nb = normalize(nb)
        ensure_header(nb)
        strip_outputs(nb)
        nbformat.write(nb, path)
        touched += 1
        print(f"Prepared: {path.name}")

    # Small manifest for reproducibility/debugging.
    manifest = {
        "count": touched,
        "notebooks": [p.name for p in paths],
    }
    (NOTEBOOKS_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
