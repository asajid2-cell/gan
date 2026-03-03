"""Deep Generative Genre Remastering (DGGR).

This repository historically had two different Python packages named `src/`:
- `lab 2/src` for Lab 2 utilities
- `lab 3/src` for Lab 3/4 utilities

For collaboration and packaging, the canonical code now lives in the single
top-level package `dggr/`. The original `lab 2/src` and `lab 3/src` packages
are kept as thin compatibility shims so existing run scripts still work.
"""

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
