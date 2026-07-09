# Desktop Outputs Audit - 2026-04-01

## Consolidation

All `dggr_*` output folders that were previously under `C:\Users\Ahmed\Desktop` have been moved into:

`Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs`

Inventory file:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\_audit_summary_inventory.json`

## Executive read

The project has three distinct phases now:

1. The original source-conditioned / hybrid translator family
2. The autonomous hybrid postprocess era
3. The new-model era: scratch retrieval, scratch structure diffusion, sourcehint retrieval, pretrained Encodec, retrieval+pretrained blend

The current state is:

- The **best practical perceptual path** is still the late **hybrid/source-conditioned** family, not the newest trained models.
- The **best objective style-separation path** is in the **retrieval** and **retrieval+pretrained blend** families.
- The **judge is now partially misaligned** with audible quality for the newest families.
- The key remaining problem is no longer sync. It is **instrument realism and arrangement richness**.

## What moved the project forward

### 1. Offline best.pt sweep

Reference:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_offline_optimization\offline_opt_20260330_113028\summary.json`

This was the cleanest proof that the old model could be steered to a slightly more style-forward regime without collapsing realism.

Important outcome:

- `best_mid_275_b` beat `best_mid_230_a`
- tradeoff delta vs baseline: about `+0.038`
- realism drop was tiny
- style gain only improved slightly

Perceptual interpretation:

- good realism floor
- still too source-bound
- useful as the last clearly trustworthy baseline

### 2. Hybrid vocal-preserve + sync-safe backing fixes

References:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_hybrid_vocal_auto_best\hybrid_auto_best_20260331_112800_stylepush\summary.json`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_hybrid_selected_pack\selected_20260331_154010\summary.json`

This phase solved the biggest workflow problems:

- vocals became usable again
- sync stopped being the main blocker
- hybrid postprocessing made the pipeline more stable

Perceptual interpretation:

- still not enough genre separation
- instrumentation often remained filtered, thin, or warbly
- but this is still the most usable overall family

## What failed perceptually

### 3. Broad fine-tuning runs

Representative runs:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_downloads_best_finetune\run_20260328_185852`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_stage1_warble_mixed\run_20260330_044057`

Perceptual failure mode:

- underwater
- windy
- over-smoothed
- less recognizable despite better losses

Conclusion:

- broad fine-tuning against continuity / crackle / stability proxies was destructive
- these runs should not be treated as promising branches

### 4. Scratch structure diffusion family

Reference:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_structure_suite\suite_20260331_205731\summary.json`

This family looks strong numerically and weak conceptually.

Key metrics:

- `mean_target_conf`: `0.9943`
- `mean_target_margin`: `0.9920`
- `mean_fullness`: `0.7949`
- `mean_structure`: `0.0505`
- `mean_separation`: `0.0`

Audit conclusion:

- the judge is being gamed here
- target confidence is nearly saturated
- separation is literally zero
- structure is weak to mixed
- this is not evidence of a genuinely better accompaniment generator

This family should be considered **rejected** despite the attractive headline numbers.

### 5. Retrieval + pretrained blend

Reference:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_pretrained_blend_production\retrieval_pretrained_blend_20260401_004523\summary.json`

Key metrics:

- `mean_target_conf`: `0.4509`
- `mean_target_margin`: `0.0937`
- `mean_fullness`: `0.5909`
- `mean_structure`: `0.0697`
- `mean_warble`: `0.0095`
- `mean_separation`: `0.1793`

Why this looked good on paper:

- higher target confidence than the older retrieval-only production
- better target margin
- better inter-target separation
- low artifact score

Why it does not hold up perceptually:

- the music stops sounding like the original song
- vocals stop feeling naturally matched to the accompaniment
- instruments sound tinny rather than fuller
- the generator is moving style space more than it is generating convincing arrangements

Audit conclusion:

- this branch improved the judge
- but it regressed the actual listening objective
- it should **not** replace the hybrid/source-conditioned production path

## What remains the strongest branch

### 6. Scratch retrieval family

References:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_retrieval_suite\suite_20260331_214339\final_training_report.md`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_fusion_production\retrieval_prod_20260331_232340\summary.json`

Why it mattered:

- it was the first new-model family that clearly attacked source timbre leakage
- it improved genre movement and separation more honestly than the scratch structure family

Why it still falls short:

- arrangements are still not full enough
- the accompaniment still does not feel like genuinely new instrumentation
- it remains weaker than the late hybrid branch on overall listenability

Audit conclusion:

- best of the genuinely new generator families
- not yet a full production replacement

## Current ranking by practical usefulness

### Tier 1: usable

1. Late hybrid/source-conditioned path with sync-safe backing and vocal preservation
2. Offline-optimized best.pt path (`best_mid_275_b` family)

### Tier 2: useful research branches, not main production

3. Scratch retrieval family
4. Aggressive pretrained Encodec family

### Tier 3: misleading or regressive

5. Retrieval + pretrained blend production
6. Scratch structure diffusion family
7. Broad fine-tune families

## Where the project actually is

The repo is no longer blocked by vocal sync. The real strategic limitation is now:

- the accompaniment generator is still acting more like a **style-colored editor** than a **new-instrument generator**
- the styles become more separable only when the model starts to lose song identity
- the judge increasingly rewards style movement and low artifact scores even when the arrangement becomes less believable

In plain terms:

- the old system preserves the song too much
- the new systems move away from the song but do not yet produce convincing replacement instrumentation

## What the audit says to do next

Do **not** promote the retrieval+pretrained blend path as the main production system.

Use the hybrid/source-conditioned family as the practical baseline while treating the new-model families as research branches until a new model beats it perceptually on:

- realism
- arrangement fullness
- target separation
- and song identity retention at the same time

## Key folders to inspect

Practical baseline era:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_offline_optimization\offline_opt_20260330_113028`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_hybrid_vocal_auto_best`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_hybrid_selected_pack`

New-model era:

- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_retrieval_suite\suite_20260331_214339`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_pretrained_encodec_aggressive_suite\suite_20260401_001640`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_fusion_production\retrieval_prod_20260331_232340`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_pretrained_blend_production\retrieval_pretrained_blend_20260401_004523`
- `Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_structure_suite\suite_20260331_205731`

## Final audit judgment

The project has made real progress on:

- sync
- stability
- vocal usability
- objective style separation

But it has **not** yet solved the core perceptual problem:

- generating accompaniment that sounds like genuinely new, full, realistic instrumentation while still clearly belonging to the original song.

That remains the active bottleneck.
