# Perceptual Generation Audit - 2026-04-01

## Goal

Audit the major generation families with the strongest available automatic scoring stack and answer two practical questions:

1. Which result families are actually worth keeping?
2. Which recent trained-model branches should be treated as regressions even if some metrics improved?

This audit uses the repo's strongest current automatic tools:

- unified genre judge from [score_production_packs_with_genre_judge.py](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\lab 3.4\scripts\score_production_packs_with_genre_judge.py)
- realism / structure / warble / fullness summaries from the saved production reports
- previous hybrid optimization summaries

## Main conclusion

Your read is correct:

- the **newer trained-model branches did not produce a clearly better practical production path**
- the **older hybrid/source-conditioned branch is still structurally better and more listenable**
- but the old branch is still **stylistically too similar across genres**

So the project is split:

- **old hybrid path**: better song fit, better structure, better workflow
- **new model path**: better genre movement on paper, but weaker perceptually

## Strongest available automatic evidence

### Old hybrid/source-conditioned production

References:

- [offline_opt_20260330_113028/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_offline_optimization\offline_opt_20260330_113028\summary.json)
- [hybrid_auto_best_20260331_112800_stylepush/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_hybrid_vocal_auto_best\hybrid_auto_best_20260331_112800_stylepush\summary.json)
- [selected_20260331_154010/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_hybrid_selected_pack\selected_20260331_154010\summary.json)

What it is good at:

- structure preservation
- sync-safe vocals
- staying recognizably like the song
- practical listenability

What it is bad at:

- target styles remain too similar
- accompaniment often sounds filtered rather than re-authored
- style movement is present but weaker than desired

Audit status:

- **KEEP as the practical baseline**

### Scratch retrieval family

References:

- [suite_20260331_214339/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_retrieval_suite\suite_20260331_214339\summary.json)
- [retrieval_prod_20260331_232340/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_fusion_production\retrieval_prod_20260331_232340\summary.json)
- [realism_compare.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_fusion_production\retrieval_prod_20260331_232340\realism_compare.json)

Suite-level means:

- `mean_target_conf`: `0.4410`
- `mean_target_margin`: `0.0698`
- `mean_warble`: `0.0090`
- `mean_fullness`: `0.5872`
- `mean_structure`: `0.0754`

Production realism comparison against the older baseline:

- production fullness: `0.6108`
- older baseline fullness: `0.7526`
- production structure: `0.2489`
- older baseline structure: `0.2452`
- production warble: `0.0092`
- older baseline warble: `0.0062`

Interpretation:

- this is the best of the truly new model families
- it improved genre movement honestly
- but it still did not generate fuller, more convincing instrumentation than the old hybrid path

Audit status:

- **KEEP as the strongest research branch**
- **DO NOT promote over the hybrid baseline**

### Sourcehint retrieval family

Reference:

- [suite_20260331_230608/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_sourcehint_suite\suite_20260331_230608\summary.json)

Suite-level means:

- `mean_target_conf`: `0.3333`
- `mean_target_margin`: `-0.1089`
- `mean_warble`: `0.0113`
- `mean_fullness`: `0.6296`
- `mean_structure`: `0.0997`

Interpretation:

- decent middle ground
- more song-like than some later branches
- still not enough target separation
- too weak to justify as the lead branch

Audit status:

- **KEEP only as a secondary reference branch**

### Aggressive pretrained Encodec family

Reference:

- [suite_20260401_001640/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_pretrained_encodec_aggressive_suite\suite_20260401_001640\summary.json)

Suite-level means:

- `mean_target_conf`: `0.3866`
- `mean_target_margin`: `-0.0663`
- `mean_warble`: `0.0246`
- `mean_fullness`: `0.6250`
- `mean_structure`: `0.0473`

Interpretation:

- some targets improved on genre confidence
- warble was worse than the scratch retrieval family
- structure was weak
- baroque and lofi were especially poor

Audit status:

- **DROP as a main candidate**

### Retrieval + pretrained blend production

References:

- [retrieval_pretrained_blend_20260401_004523/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_pretrained_blend_production\retrieval_pretrained_blend_20260401_004523\summary.json)
- [judge_compare_vs_retrieval/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_retrieval_pretrained_blend_production\retrieval_pretrained_blend_20260401_004523\judge_compare_vs_retrieval\summary.json)

Blend production summary:

- `mean_target_conf`: `0.4509`
- `mean_target_margin`: `0.0937`
- `mean_fullness`: `0.5909`
- `mean_structure`: `0.0697`
- `mean_warble`: `0.0095`
- `mean_separation`: `0.1793`

Judge compare vs retrieval-only:

- blend conf: `0.3785` vs retrieval `0.3708`
- blend margin: `-0.0506` vs retrieval `-0.0803`
- blend separation: `0.1103` vs retrieval `0.1060`

Interpretation:

- on the judge, this looks like an improvement
- by perceptual read, it is a regression:
  - less like the song
  - worse vocal/accompaniment fit
  - thinner / tinny instrumentation
  - stronger style movement without convincing musical body

Audit status:

- **DROP as a production path**
- **Important sign that the judge is misaligned**

### Scratch structure diffusion family

Reference:

- [suite_20260331_205731/summary.json](Z:\328\CMPUT328-A2\codexworks\301\414-pl1\Desktop Outputs\dggr_per_genre_structure_suite\suite_20260331_205731\summary.json)

Suite-level means:

- `mean_target_conf`: `0.9943`
- `mean_target_margin`: `0.9920`
- `mean_warble`: `0.0271`
- `mean_fullness`: `0.7949`
- `mean_structure`: `0.0505`

Interpretation:

- almost certainly a metric illusion
- target confidence is saturated
- separation is effectively nonexistent inside the family
- structure is too weak relative to the headline score
- this is not a trustworthy winner

Audit status:

- **DROP**
- **Treat as judge gaming, not real success**

## Filtered keep/drop result

### Keep

1. **Late hybrid/source-conditioned family**
   - Best practical listening baseline
   - Best structure/song fit
   - Still the branch to compare everything against

2. **Scratch retrieval family**
   - Best genuinely new model family
   - Real style movement improvement
   - Still not a production replacement

3. **Sourcehint retrieval family**
   - Secondary reference only
   - More balanced than some later branches, but not strong enough overall

### Drop

1. **Broad fine-tune families**
   - Over-smoothed
   - Underwater/windy failures

2. **Scratch structure diffusion**
   - Not believable as a real winner
   - Metrics are not trustworthy here

3. **Aggressive pretrained Encodec**
   - Worse warble and weak structure

4. **Retrieval + pretrained blend**
   - Judge win, perceptual loss
   - Not acceptable as main production

## Where things actually stand

The repo has not yet produced a new model that beats the older hybrid pipeline on the full real objective.

Current best practical statement:

- **older hybrid path**: more coherent and more song-faithful
- **new retrieval family**: more style movement, but still not fuller or more believable enough

So your summary is basically right:

- newer models mostly did not pan out
- older ones are structurally better
- older ones are still stylistically too similar

## Most important conceptual finding

The current judge is good enough to detect some artifact differences and some target movement, but it is still too willing to reward:

- style movement without believable arrangement
- target confidence without enough song identity
- low warble without enough musical body

That is why some recent branches looked good numerically and bad by ear.

## Practical recommendation from this audit

For now:

- use the late hybrid/source-conditioned family as the production baseline
- keep scratch retrieval as the only new-model family worth further work
- treat the other newer trained-model branches as filtered out

If we continue model work, the target should be:

- fuller accompaniment generation
- stronger instrumentation richness
- no loss of structure/song identity
- and a judge recalibrated to penalize tinny or de-bodied outputs more strongly
