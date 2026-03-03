"""Compatibility shim for legacy imports.

Historically, Lab 3 code was imported as `from src.<module> import ...` by adding
`lab 3/` to `sys.path`. The canonical code now lives in the top-level `dggr/`
package.

This shim keeps existing imports working by ensuring the repo root is on
`sys.path`, so `import dggr` succeeds even when running from inside `lab 3/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
