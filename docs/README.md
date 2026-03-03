# Documentation (DGGR)

This `docs/` tree is organized using the **Diataxis** documentation framework:
- `docs/tutorials/`: learning-oriented walkthroughs (do this, then that).
- `docs/howto/`: goal-oriented recipes (how to train, how to export samples).
- `docs/reference/`: facts and interfaces (CLI flags, data formats, metrics).
- `docs/explanation/`: design rationale and technical background.

Why this structure:
- Diataxis explicitly separates *what a user is trying to do* (tutorial vs how-to) from *what they need to know* (reference vs explanation).
- This prevents the common failure mode where docs mix narrative, procedure, and spec details in the same page.

Attributions and templates used:
- `docs/references.md` (Diataxis, GitHub template docs, Contributor Covenant).

Start here:
- `docs/tutorials/01_quickstart_longform.md`
- `docs/howto/reproduce_best_runs.md`
- `docs/explanation/architecture.md`

Recommended next reads:
- `docs/reference/cli.md`
- `docs/reference/data_formats.md`
- `docs/reference/env_vars.md`
- `docs/explanation/codec_vs_diffusion.md`
- `docs/explanation/longform_coherence.md`
