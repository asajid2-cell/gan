# Pipeline Diagnosis: What Is Actually Missing?

This note is the clean-slate diagnosis for DGGR after reviewing Labs 1 through 4, the saved codec runs, the diffusion runs, and the realism-supervisor outputs.

## Bottom line

The current pipeline is not primarily failing because Lab 1 or Lab 2 are broken.

It is failing because Lab 3 is trying to solve a "structural genre remastering" problem with objectives and architecture that are mostly suited to high-quality source-conditioned editing.

That mismatch produces the main pattern seen across the repo:

- content is preserved
- the audio remains plausible enough to pass shallow gates
- but the result often does not truly sound like it belongs to the target genre

## What is working

### Lab 1

Lab 1 is already doing its intended job:

- `z_content` is style-suppressed enough to preserve melody/structure
- `z_style` is style-informative
- the music gate is usable

The checkpoint and audit results show this stage is not the dominant bottleneck.

### Lab 2

Lab 2 also appears directionally correct:

- the target vector space is separable
- centroids exist and are stable enough to use as conditioning

So the project does have a usable conditioning signal.

## What is not working

### 1. The label space is weaker than the research question

The current 4-way genre buckets are still closely tied to dataset provenance:

- `baroque_classical`
- `hiphop_xtc`
- `lofi_hh_lfbb`
- `cc0_other`

These are useful engineering buckets, but they are not yet a rich representation of genre in the musical sense.

This creates a problem:

- it is easier for a model to learn dataset-associated textures than to learn what makes a song feel as if it was composed in a different genre

So even when the system learns a measurable style signal, it can still miss the actual remastering objective.

### 2. The old Lab 3 gate overstates success

The strongest codec runs achieved excellent internal metrics such as:

- high `style_conf`
- high `style_acc`
- high `mps`

But the realism supervisor tells a different story on generated audio:

- target-style accuracy often collapses toward chance
- target-style cosine is weak or negative
- realism and style movement do not improve together

This means the original gate is partly measuring whether the model can satisfy a local classifier, not whether the generated waveform actually lives in the target genre manifold.

### 3. The codec branch is structurally biased toward editing, not reconstruction

The codec translator is powerful as a latent editor, but the overall branch still behaves like:

- source waveform
- frozen codec embedding
- learned latent edit
- decode back to waveform

That is a strong design for "change the source without destroying it."

It is not automatically a strong design for:

- rewriting instrumentation
- rewriting arrangement feel
- changing groove identity
- replacing the entire acoustic environment

That is the core conceptual gap.

Even the strongest direct-output runs are still constrained by a source-conditioned codec manifold.

### 4. The training objective still rewards staying near the source

Across the codec branch, late stages still retain strong source-preservation pressure:

- latent similarity
- continuity to source latents
- waveform spectral similarity to the source
- content-preservation losses

These are all useful individually.

But together they create a strong bias toward:

- minimal safe edits

That is why many runs either:

- sound realistic but barely move

or:

- move more but become harsh, codec-artifact-heavy, or unstable

### 5. Long-form coherence is downstream, not the root bottleneck

Lab 4 adds useful overlap locking and drift control.

But if the short-form generator is not already producing convincing target-style chunks, Lab 4 cannot fix that.

It can only:

- make bad short chunks more coherent over time

So the long-form work should remain downstream of short-form quality, not a parallel source of truth.

## What the saved runs suggest

The saved codec runs show a recurring pattern:

- internal gate scores can be strong
- realism-first evaluation says target-style movement is still weak

The saved diffusion runs suggest a different pattern:

- less stable training
- stronger target-style realism when the checkpoint is right
- more promise for real remastering, but also more sensitivity

This suggests the repo is already pointing toward the real answer:

- codec branch = good conservative editor
- diffusion branch = better candidate for true remastering

## Conceptually missing piece

The project wants "from-the-ground-up genre remastering."

What is missing is a stage that explicitly models:

- a stable musical plan
- a target-style realization process that is not mostly judged by closeness to the source

In other words:

- Lab 1 gives the structural skeleton
- Lab 2 gives a target style blueprint
- but Lab 3 currently does not yet have the right kind of generative rewrite mechanism to turn skeleton plus blueprint into a convincingly re-authored track

That is the missing concept.

## Practical implication

The next clean-slate cycle should be hypothesis-driven:

### Hypothesis A: the main bottleneck is still label quality

Test with:

- stronger source-balanced schemas
- auto-genre labels
- notebook-first audits of source leakage

### Hypothesis B: the codec branch cannot express large enough change

Test with:

- fresh codec runs, but evaluate them honestly against generated-audio realism and target-style movement
- treat codec as a baseline editor, not automatically the final answer

### Hypothesis C: diffusion is the correct remastering branch

Test with:

- fresh diffusion training from scratch
- better checkpoint selection using realism and style movement, not only validation loss
- short-form wins first, then long-form Lab 4

## New decision rule

Do not promote a model just because:

- `mps` is high
- internal style confidence is high

Promote a model only if it clears both:

1. realism
2. target-style movement on generated audio

That is the central reset for Lab 3.1.
