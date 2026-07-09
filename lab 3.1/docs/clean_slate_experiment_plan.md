# Clean-Slate Experiment Plan

This plan turns the pipeline diagnosis into a concrete from-scratch workflow.

## Goal

The reset target is not:

- "get high internal style confidence"

The reset target is:

- preserve musical identity
- move clearly into the target genre on generated audio
- stay perceptually realistic enough that the result does not sound robotic, warbly, or brittle

## What to keep

Keep the existing conceptual split:

1. Lab 1: structural/content representation
2. Lab 2: target-style blueprint
3. Lab 3: short-form generation
4. Lab 4: long-form coherence

The evidence does not support rebuilding Labs 1 and 2 first.

## What to reset

Reset Lab 3 training and evaluation from scratch.

That means:

- do not continue inherited codec runs
- do not promote checkpoints based on old internal gates alone
- do not move to Lab 4 until short-form generation has already passed realism plus movement

## Research hypotheses

### H1. Data/labels are partly mismatched to the research question

Current genre buckets are close to dataset provenance and may encourage source/dataset shortcuts.

Test:

- source-balanced mini training subsets
- auto-genre relabeling or clustered style buckets
- notebook audits for source leakage and per-bucket diversity

Success signal:

- target-style movement improves without the classifier simply learning source fingerprints

### H2. Codec translation is a strong editor, but not the best remastering engine

Codec runs preserve melody well, but realism and target-style movement disagree.

Test:

- treat codec as the conservative baseline
- rerun codec from scratch with honest generated-audio evaluation
- compare directly against a fresh diffusion baseline on the same fixed validation plan

Success signal:

- codec only remains primary if it can move beyond chance-level target-style accuracy on generated audio while holding realism

### H3. Diffusion is the better branch for real remastering

Diffusion is more aligned with "re-author the acoustic world from the blueprint" than codec latent editing.

Test:

- fresh short-form diffusion run from scratch
- fixed validation transfer plan
- checkpoint ranking by realism and target-style movement, not validation loss alone

Success signal:

- a diffusion checkpoint beats codec on both target-style movement and subjective listening while remaining usable for Lab 4

## New promotion rules

Do not promote a checkpoint unless it passes both:

1. realism
2. transformation

### Realism axis

- `fad_mert`
- `target_hf_mae`
- `target_dynamic_range_mae_db`
- listening audit for warble/static/brittleness

### Transformation axis

- `style_target_acc`
- `style_target_cos`
- melody preservation (`mps`)
- direct listening: can a listener identify the new genre without losing the original musical identity?

## Notebook-first sequence

### Notebook 00: pipeline audit

Purpose:

- summarize old runs
- verify that the old codec gate and the realism supervisor disagree

### Notebook 01: Lab 1 / Lab 2 audit

Purpose:

- verify Labs 1 and 2 remain good enough to reuse
- only reopen them if leakage or separability looks materially worse than recorded

### Notebook 02: codec from scratch

Purpose:

- train a fresh codec baseline from zero
- use fixed validation clips and generated-audio evaluation at every decision point

Interpretation rule:

- codec is the safety baseline, not the presumed winner

### Notebook 03: diffusion from scratch

Purpose:

- train a fresh diffusion baseline from zero
- evaluate early and often on the same validation clips

Interpretation rule:

- if diffusion wins on generated-audio target-style movement and listening, it becomes the main Lab 3 branch

### Notebook 04: long-form validation

Purpose:

- only after short-form quality is already acceptable
- evaluate overlap, drift, and prefix-lock coherence on the chosen short-form branch

## Minimum experiment set before another long run

1. Refresh the audit tables.
2. Lock a fixed validation transfer set.
3. Run one fresh codec baseline from scratch.
4. Run one fresh diffusion baseline from scratch.
5. Compare them under the same realism plus movement criteria.
6. Only then authorize a longer production run.

## Expected outcome

The likely outcome of this reset is:

- Labs 1 and 2 are reused
- codec becomes the conservative baseline/editor path
- diffusion becomes the main candidate for true remastering
- Lab 4 is only applied after one branch proves itself in short-form
