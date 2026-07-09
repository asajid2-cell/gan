# DGGR Project State - 2026-04-01

This page is the current high-level state of the project across Labs 1-4 and the post-lab experimental work.

It is meant to answer:

- what each stage accomplished
- what actually worked
- what failed
- what the current production baseline is
- what the main remaining bottlenecks are

Recent supporting audits:

- [desktop_outputs_audit_20260401.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/desktop_outputs_audit_20260401.md)
- [perceptual_generation_audit_20260401.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/perceptual_generation_audit_20260401.md)
- [generation_family_filter_20260401.json](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/generation_family_filter_20260401.json)

## Executive Summary

The project has succeeded at:

- learning usable content/style representations
- building a separable target-style vector space
- training a short-form diffusion accompaniment editor with a real realism floor
- stabilizing long-form generation enough for practical hybrid vocal-preserve workflows
- restoring vocal clarity and sync to a usable level in the hybrid path

The project has **not** yet succeeded at:

- generating obviously new, full, believable accompaniment instrumentation
- producing strong genre separation without losing song identity
- replacing the best hybrid/source-conditioned path with a clearly superior new model

Current state in one sentence:

- the best practical production path is still the late hybrid/source-conditioned branch, while the best new-model branch is the scratch retrieval family, which improves style movement but still does not beat the hybrid branch on overall listenability.

## Timeline

### Lab 1: Deconstruction Encoder

Purpose:

- learn `z_content`
- learn `z_style`
- train a reliable music gate

Outcome:

- this stage worked
- the representation is strong enough to support later conditioning and auditing
- Lab 1 remains a real success, not a bottleneck

Historical metrics:

- `style_probe_accuracy ~= 0.94`
- `content_leakage_above_baseline ~= 0.11`
- `gate_roc_auc ~= 0.93`

Relevant docs:

- [lab 1/README.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%201/README.md)
- [lab1_deconstruction_encoder.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%201/docs/lab1_deconstruction_encoder.md)

### Lab 2: Target Vector Space

Purpose:

- turn Lab 1 into a style-harvesting system
- build genre centroids / target vectors
- validate style-space separability

Outcome:

- this stage also worked
- the target vector space is meaningfully separable
- Lab 2 is not the main reason current outputs are too similar

Historical metrics:

- `silhouette ~= 0.49`
- `linear_probe_acc ~= 0.86`
- `nearest_centroid_acc ~= 0.85`

Important interpretation:

- the style space is usable
- later style-collapse is mostly not because Lab 2 completely failed
- later failures are more about the generator and objective than the centroid construction itself

Relevant docs:

- [lab 2/README.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%202/README.md)
- [lab2_target_vector_space.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%202/docs/lab2_target_vector_space.md)

### Lab 3: Reconstruction / Codec / Diffusion

This stage split into two major tracks:

- codec-latent transfer
- diffusion accompaniment generation

#### Codec-latent transfer

Strengths:

- preserved content well
- achieved strong style metrics on paper
- was a useful stepping stone in understanding style control

Weaknesses:

- realism and timbral quality became the real bottleneck
- later rebuild tuning and realism supervision were required

#### Diffusion branch

This became the more important practical branch.

Best historical checkpoint family:

- `run_d002`

Important finding:

- later epochs improved numeric loss
- perceptual quality peaked earlier
- the diffusion model was usable in short form, but long-form rollout exposed warble/static/control issues

What diffusion solved:

- it produced the best practical accompaniment editor the repo has had so far
- it set the realism floor for later long-form and hybrid work

What diffusion did not solve:

- strong style re-authoring
- obviously different instrumentation across targets
- robust long-form behavior without additional control logic

Relevant docs:

- [lab 3/README.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%203/README.md)
- [lab3_diffusion.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%203/docs/lab3_diffusion.md)
- [codec_vs_diffusion.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%203/docs/codec_vs_diffusion.md)
- [05_realism_supervisor.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%203/docs/05_realism_supervisor.md)

### Lab 4: Long-form coherence

Purpose:

- extend short-form generation to full-song or long-horizon generation
- control seam continuity, drift, re-anchoring, and chunk interactions

Outcome:

- hard seam issues became measurable and manageable
- the remaining perceptual issues shifted toward:
  - warble
  - static / harshness
  - vocal instability
  - style suppression from conservative long-form controls

Key conceptual result:

- long-form generation was not mainly failing because of visible seam breaks
- it was failing because recursive rollout compounded local artifacts and conservative anchoring suppressed style movement

Relevant docs:

- [lab 4/README.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%204/README.md)
- [lab4_longform.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%204/docs/lab4_longform.md)
- [longform_coherence.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%204/docs/longform_coherence.md)
- [longform_tuning.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%204/docs/longform_tuning.md)

## Post-Lab Experimental Era

After the original lab sequence, the repo moved into a heavier experimental phase.

That phase produced three main eras:

1. offline inference optimization on `run_d002`
2. hybrid vocal-preserve production engineering
3. new-model research families

### 1. Offline inference optimization

This was the cleanest proof that the old diffusion model still had some headroom.

Key run:

- [offline_opt_20260330_113028](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_offline_optimization/offline_opt_20260330_113028)

Important result:

- `best_mid_275_b` beat the safer baseline `best_mid_230_a`
- style gain improved slightly
- realism barely regressed

Interpretation:

- the old diffusion baseline was not tapped out
- but the gain was modest
- this path improved tradeoff tuning, not generator capacity

### 2. Hybrid vocal-preserve and sync-safe production path

This is the branch that made the project practically usable again.

What it did:

- separated vocals and accompaniment
- preserved vocals more directly
- fixed major sync failures
- added backing alignment, debleed, and postprocess corrections
- enabled automatic target-specific preset selection

Best practical output roots:

- [dggr_hybrid_vocal_auto_best](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_hybrid_vocal_auto_best)
- [dggr_hybrid_selected_pack](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_hybrid_selected_pack)

What this branch solved:

- vocals became clear enough
- sync stopped being the dominant blocker
- long-form outputs became much more listenable

What it still does poorly:

- accompaniment often sounds filtered instead of newly authored
- target genres are still too similar
- the model behaves more like a high-quality remastering editor than a true re-arranger

This remains the current practical baseline.

### 3. New-model research families

The goal here was to break out of the old source-conditioned ceiling.

The main families tried were:

- broad fine-tunes from the old diffusion base
- scratch retrieval-conditioned accompaniment models
- scratch structure diffusion
- sourcehint retrieval
- pretrained Encodec fusion
- retrieval + pretrained blend
- retrieval body-style variant

## What Worked and What Failed

### Keep: late hybrid/source-conditioned production

Verdict:

- keep as the production baseline

Why:

- best song fit
- best practical listenability
- best vocal/accompaniment usability
- strongest overall workflow

Weakness:

- still too conservative stylistically

References:

- [selected_20260331_154010](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_hybrid_selected_pack/selected_20260331_154010)

### Keep: offline optimized best.pt path

Verdict:

- keep as a clean non-hybrid reference

Why:

- clear proof of controlled style-vs-realism tradeoff improvement
- useful baseline for evaluating whether later work actually improved anything

### Keep as research: scratch retrieval family

Verdict:

- best genuinely new model family
- not yet a production replacement

Why it mattered:

- it attacked source timbre leakage more honestly than the old translator
- it improved style movement and genre confidence more than the hybrid baseline

Why it still fails:

- accompaniment still does not sound full enough
- outputs often feel thinner than the hybrid baseline
- arrangement richness is still insufficient

Main roots:

- [suite_20260331_214339](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_per_genre_retrieval_suite/suite_20260331_214339)
- [retrieval_prod_20260331_232340](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_retrieval_fusion_production/retrieval_prod_20260331_232340)

### Secondary reference: sourcehint retrieval

Verdict:

- keep only as a reference branch

Why:

- somewhat balanced
- not strong enough on style separation to lead

### Rejected: broad fine-tuning families

Verdict:

- drop

Representative roots:

- [dggr_downloads_best_finetune](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_downloads_best_finetune)
- [diffusion_stage1_warble_mixed](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%203.3/outputs/diffusion_stage1_warble_mixed)

Failure mode:

- underwater
- windy
- over-smoothed
- less recognizable despite lower losses

Conclusion:

- these runs over-optimized the wrong proxies

### Rejected: scratch structure diffusion

Verdict:

- drop

Why:

- extremely strong headline metrics
- weak believable structure
- effectively zero meaningful intra-family separation
- likely judge gaming rather than genuine quality

Root:

- [suite_20260331_205731](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_per_genre_structure_suite/suite_20260331_205731)

### Rejected: aggressive pretrained Encodec

Verdict:

- drop as a lead branch

Why:

- worse warble than retrieval
- weak structure
- some targets regressed badly

### Rejected: retrieval + pretrained blend

Verdict:

- drop as production path

Why:

- improved the judge
- regressed perceptually
- instrumentation became tinny
- song fit and vocal/accompaniment cohesion got worse

Root:

- [retrieval_pretrained_blend_20260401_004523](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_retrieval_pretrained_blend_production/retrieval_pretrained_blend_20260401_004523)

### New long-run body-style retrieval variant

This branch was trained to address the short retrieval family’s thinness by explicitly adding:

- body/fullness loss
- groove/envelope loss
- stronger judge-based checkpoint selection

Latest serious run:

- [suite_20260401_102538](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_per_genre_retrieval_body_style_suite/suite_20260401_102538)

Compared with the earlier retrieval suite, it improved:

- overall score
- target confidence
- target margin
- warble
- fullness

But it still lost slightly on structure.

Current interpretation:

- this is a better research branch than the original scratch retrieval suite
- it is still not enough to unseat the hybrid baseline on overall practical quality

## Current Best Paths

### Best practical production path

Use the late hybrid/source-conditioned family with:

- vocal preservation
- sync-safe backing
- target-specific backing postprocess
- debleed and alignment fixes

Best root:

- [selected_20260331_154010](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_hybrid_selected_pack/selected_20260331_154010)

### Best new-model research path

Use the retrieval family, now preferably the longer body-style retrieval variant for future research iterations:

- [suite_20260401_102538](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_per_genre_retrieval_body_style_suite/suite_20260401_102538)

## Current Bottlenecks

The project is no longer mainly blocked by:

- vocal sync
- obvious seam breaks
- total instability

The actual remaining bottlenecks are:

1. accompaniment realism
2. arrangement richness / fullness
3. obvious genre separation without losing song identity
4. judge alignment with perceptual quality

In plain terms:

- the hybrid branch preserves the song too much
- the newer generators move farther from the song but still do not produce convincing replacement instrumentation

## What We Know Now

1. Lab 1 and Lab 2 basically worked.
   They are not the main failure point anymore.

2. The old diffusion branch is still the best realism anchor.
   It gives the strongest perceptual floor, especially once wrapped in hybrid controls.

3. Broad fine-tuning against stability-like losses was destructive.
   Lower loss did not mean better audio.

4. The current judge is useful but not fully aligned.
   It still over-rewards:
   - style movement without enough believable arrangement
   - low artifacts without enough body
   - target confidence without enough song fit

5. The best newer family is retrieval, not pretrained blend and not scratch structure diffusion.

6. Longer training matters.
   Very short probes are enough to reject bad ideas, but not enough to expect major perceptual gains.

## Recommended Reading Order

If someone is joining the repo fresh:

1. [architecture.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/architecture.md)
2. [results.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/results.md)
3. this page
4. [lab3_diffusion.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%203/docs/lab3_diffusion.md)
5. [lab4_longform.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/lab%204/docs/lab4_longform.md)
6. [desktop_outputs_audit_20260401.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/desktop_outputs_audit_20260401.md)
7. [perceptual_generation_audit_20260401.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/perceptual_generation_audit_20260401.md)

## Artifact Roots

Repo-local archive of moved experimental outputs:

- [Desktop Outputs](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs)

Most important subfolders:

- [dggr_offline_optimization](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_offline_optimization)
- [dggr_hybrid_vocal_auto_best](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_hybrid_vocal_auto_best)
- [dggr_hybrid_selected_pack](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_hybrid_selected_pack)
- [dggr_per_genre_retrieval_suite](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_per_genre_retrieval_suite)
- [dggr_per_genre_retrieval_body_style_suite](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_per_genre_retrieval_body_style_suite)
- [dggr_retrieval_fusion_production](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_retrieval_fusion_production)
- [dggr_retrieval_pretrained_blend_production](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/Desktop%20Outputs/dggr_retrieval_pretrained_blend_production)

## Bottom Line

The project has made real progress on:

- representations
- target style vectors
- short-form realism
- long-form stability
- hybrid vocal usability

The project has not yet solved:

- full believable accompaniment re-authoring
- strong style distinctness with preserved song identity

So the current best truthful summary is:

- use the hybrid/source-conditioned branch in practice
- keep the retrieval body-style branch as the best current new-model research direction
- treat the remaining newer families as rejected unless they later beat the hybrid baseline perceptually
