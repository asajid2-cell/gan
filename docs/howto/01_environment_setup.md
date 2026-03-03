# Environment Setup (How-To)

This repo is intentionally light on packaging so collaborators can use their preferred setup.

Recommended workflow:

1. Create a Python environment (venv/conda).
2. Install dependencies from `requirements.txt`.
3. Run entrypoints from the repo root so relative paths work.

Notes:
- Large artifacts and caches are ignored by git (`saves/`, `saves2/`).
- Some notebooks/scripts will download models via HuggingFace (e.g., BigVGAN, MERT).

