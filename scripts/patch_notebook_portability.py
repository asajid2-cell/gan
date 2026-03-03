from __future__ import annotations

from pathlib import Path
import re

import nbformat
from nbformat.validator import normalize


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


SETUP_CELL = """from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for _ in range(8):
        if (p / "dggr").exists():
            return p
        p = p.parent
    return (start or Path.cwd()).resolve()


REPO_ROOT = _find_repo_root()
DATA_ROOT = Path(os.environ.get("DGGR_DATA_ROOT", str(REPO_ROOT / "data")))
MANIFESTS_ROOT = Path(os.environ.get("DGGR_MANIFESTS_ROOT", str(DATA_ROOT / "_lab1_manifests")))

print("REPO_ROOT:", REPO_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("MANIFESTS_ROOT:", MANIFESTS_ROOT)
"""


REPLACEMENTS = [
    (
        'DATA_ROOT = Path(r"Z:\\\\DataSets")',
        "# DATA_ROOT is set in the portability setup cell above",
    ),
    (
        'DATA_ROOT = Path(r"Z:\\DataSets")',
        "# DATA_ROOT is set in the portability setup cell above",
    ),
    (
        '_SCRIPT_DIR = Path(r"Z:\\328\\CMPUT328-A2\\codexworks\\301\\414-pl1\\lab 3")',
        '_SCRIPT_DIR = REPO_ROOT / \"lab 3\"',
    ),
    (
        'BASE = Path(r"Z:\\328\\CMPUT328-A2\\codexworks\\301\\414-pl1")',
        "BASE = REPO_ROOT",
    ),
]

_RE_Z_DATASETS_MANIFESTS = re.compile(
    r"Path\(\s*r?['\"]Z:(?:/|\\\\)DataSets(?:/|\\\\)_lab1_manifests(?:/|\\\\)(?P<rest>[^'\"]+)['\"]\s*\)"
)
_RE_Z_DATASETS = re.compile(
    r"Path\(\s*r?['\"]Z:(?:/|\\\\)DataSets(?:/|\\\\)(?P<rest>[^'\"]+)['\"]\s*\)"
)


def ensure_setup_cell(nb: nbformat.NotebookNode) -> None:
    # We already insert a general header markdown cell via scripts/prepare_notebooks.py.
    # Add a dedicated portability setup cell right after it.
    if len(nb.cells) >= 2:
        c1 = nb.cells[1]
        if c1.get("cell_type") == "code" and "REPO_ROOT" in (c1.get("source") or "") and "DGGR_DATA_ROOT" in (c1.get("source") or ""):
            return
    nb.cells.insert(1, nbformat.v4.new_code_cell(SETUP_CELL))


def patch_sources(nb: nbformat.NotebookNode) -> int:
    changed = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        new = src
        for old, rep in REPLACEMENTS:
            if old in new:
                new = new.replace(old, rep)

        # Rewrite absolute Z:/DataSets paths inside code into portable variables.
        def _norm_rest(rest: str) -> str:
            # Convert any backslashes to forward slashes for Path / "a/b" style.
            return rest.replace("\\", "/").lstrip("/")

        def _repl_manifest(m: re.Match) -> str:
            rest = _norm_rest(m.group("rest"))
            return f'MANIFESTS_ROOT / "{rest}"'

        def _repl_data(m: re.Match) -> str:
            rest = _norm_rest(m.group("rest"))
            return f'DATA_ROOT / "{rest}"'

        new = _RE_Z_DATASETS_MANIFESTS.sub(_repl_manifest, new)
        new = _RE_Z_DATASETS.sub(_repl_data, new)

        if new != src:
            cell["source"] = new
            changed += 1
    return changed


def main() -> int:
    if not NOTEBOOKS_DIR.exists():
        raise SystemExit(f"Missing notebooks dir: {NOTEBOOKS_DIR}")

    paths = sorted(p for p in NOTEBOOKS_DIR.glob("*.ipynb") if p.is_file())
    if not paths:
        print("No notebooks found.")
        return 0

    for path in paths:
        nb = nbformat.read(path, as_version=4)
        _, nb = normalize(nb)
        ensure_setup_cell(nb)
        n = patch_sources(nb)
        nbformat.write(nb, path)
        if n:
            print(f"Patched: {path.name} ({n} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
