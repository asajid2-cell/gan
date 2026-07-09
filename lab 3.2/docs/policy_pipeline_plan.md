# Long-Form Policy Pipeline

This folder turns the long-form tuning problem into a searchable, trainable policy problem.

## Core idea

One fixed long-form setting does not generalize across songs. The policy pipeline therefore treats long-form generation as a selection problem:

1. pick a checkpoint
2. pick a long-form setting preset
3. render the output
4. score the result
5. train a model to predict which checkpoint/preset pair should work best for a new song

The diffusion model still generates audio. The policy only decides how to drive it.

## What is being ranked

Each generated long-form output is scored with a proxy objective that favors:

- low seam error
- low crackle / static / blown-out high-frequency behavior
- preserved musical content
- some style pressure

The current score uses:

- `boundary_mel_mse_mean`
- `boundary_disc_db_mean`
- generated high-frequency roughness
- generated spectral flatness
- clipping fraction
- high-frequency ratio drift from source
- vocal-band ratio drift from source
- dynamic-range drift from source
- source/generated chroma cosine
- source/generated onset correlation
- a style-intent prior derived from the sampler settings

The score is intentionally realism-first. The goal is to find stable, non-staticky outputs first, then push style further once that base is reliable.

## Training target

The policy is trained on existing sweep outputs. Each row in the training table is:

- one source excerpt
- one checkpoint
- one setting preset
- one resulting long-form output
- one scalar proxy score

The model learns:

`source features + target genre + checkpoint id + setting values -> predicted score`

At inference time, the policy evaluates many candidate checkpoint/preset combinations for a new song and selects the highest-scoring one.

## Model choice

The initial policy model is a `RandomForestRegressor`.

Reasons:

- nonlinear enough to model song/setting interactions
- easy to train on a modest dataset
- robust with mixed numeric and one-hot encoded features
- fast enough for iterative use

## Evaluation

The pipeline uses grouped train/test splitting by source excerpt so the evaluation reflects generalization to new songs rather than memorizing repeated settings on the same song.

It reports:

- fit metrics (`MAE`, `R^2`)
- policy-selected mean true score on held-out songs
- oracle best mean true score
- best fixed combo mean true score
- best style-forward fixed combo mean true score

## Demo generation

For fresh songs, the pipeline renders:

- `policy_top1`
- `baseline_safe`
- `baseline_style`

This makes it possible to listen to:

1. a safe fixed recipe
2. a stronger fixed style recipe
3. the policy-selected recipe

and compare whether the policy is actually improving the tradeoff.
