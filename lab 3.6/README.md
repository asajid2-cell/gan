# Lab 3.6 Piano Arranger

This workspace is for the new song-to-piano arrangement model. It is separate from the existing real-music transfer checkpoint. The older DGGR transfer path may be used as a baseline, but the goal here is to train a model that emits piano performance structure: notes, voicings, velocity, sustain, and renderable MIDI/WAV artifacts.

Read first:

```text
docs/PIANO_ARRANGER_INSTRUCT.md
```

## Target Pipeline

Planned entry point:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action discover-piano
python "lab 3.6\run_piano_arranger_pipeline.py" --action cache --max-tracks 64 --cache-seconds 8
python "lab 3.6\run_piano_arranger_pipeline.py" --action heuristic-baseline --source-audio "<song.wav>" --seconds 30
python "lab 3.6\run_piano_arranger_pipeline.py" --action train --cache-dir "saves2\piano_arranger\cache\bootstrap_pseudo_v1"
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer --checkpoint "<run>\checkpoints\latest.pt" --source-audio "<song.wav>"
python "lab 3.6\run_piano_arranger_pipeline.py" --action evaluate --arrangement-json "<output.json>"
```

## Planned Artifacts

- `data/piano_arranger_manifests/`: piano candidate manifests and discovery reports.
- `saves2/piano_arranger/cache/`: source features and piano targets.
- `saves2/piano_arranger/runs/<run_id>/`: training summaries, history, checkpoints, and epoch samples.
- `saves2/piano_arranger/outputs/`: final MIDI/WAV renders.
- `saves2/piano_arranger/reports/`: baseline comparisons, automated metrics, and listening packets.

## First Build Order

1. `dggr/piano_arranger_data.py`: discover piano-heavy candidates and build the cache schema.
2. `dggr/piano_arranger_render.py`: convert piano-roll/MIDI to inspectable output. MIDI and a deterministic preview WAV renderer are available.
3. `dggr/piano_arranger_baseline.py`: source-chroma/onset heuristic baseline. It writes MIDI, JSON, summary, and optional WAV.
4. `dggr/piano_arranger_cache.py`: source conditioning and bootstrap piano-roll target tensors.
5. `dggr/piano_arranger_models.py`: first trainable piano-roll generator.
6. `dggr/piano_arranger_train.py`: smoke train, checkpoints, history, and epoch sample MIDI/WAV output.
7. `dggr/piano_arranger_infer.py`: load a checkpoint and render arbitrary source audio to MIDI/JSON/WAV.
8. `dggr/piano_arranger_eval.py`: automated structural metrics plus optional source-aware chroma/onset alignment.
9. `dggr/piano_arranger_batch.py`: paired-checkpoint validation and source-only multi-row render audits.

Training sample evaluation is enabled by default. Use `--no-sample-eval` for speed, or `--no-source-sample-eval` to skip loading source audio during sample evaluation.

## Cache Tensor Contract

The bootstrap cache currently uses the heuristic baseline as a pseudo-target until real piano transcription or MIDI data is added.

Files:

- `source_condition.npy`: `[N, 17, T]` with 12 chroma channels plus onset, RMS, beat, spectral centroid, and zero crossing rate.
- `target_onset.npy`: `[N, 88, T]` piano key onset targets.
- `target_frame.npy`: `[N, 88, T]` sustained note-frame targets.
- `target_velocity.npy`: `[N, 88, T]` normalized velocity targets.
- `target_pedal.npy`: `[N, T]` normalized sustain pedal target.
- `target_density.npy`: `[N, 2, T]` onset-density and frame-density planning targets.
- `target_register.npy`: `[N, 3, T]` low/mid/high register-activity planning targets.
- `target_chord.npy`: `[N, 13, T]` active pitch-class set plus no-chord planning target.
- `target_bass.npy`: `[N, 13, T]` lowest active pitch-class plus no-bass planning target.
- `target_voicing.npy`: `[N, 4, T]` note-count, span, center, and high-register fraction planning target.
- `target_event.npy`: `[N, 4, T]` onset-density, note-off-density, frame-change, and chord-change planning target.
- `target_pc_onset.npy`: `[N, 12, T]` pitch-class onset planning target.
- `target_role.npy`: `[N, 5, T]` bass activity, chord-frame activity, melody-register activity, normalized polyphony, and active-frame velocity weight.
- `target_melody.npy`: `[N, 4, T]` high-register activity, upper-register activity, normalized top pitch, and top velocity.
- `target_texture_role.npy`: `[N, 4, T]` joint bass-floor, chord-body, inner-motion, and top-line presence targets.
- `target_section_role.npy`: `[N, 4, T]` section bass continuity, chord body, melody presence, and fullness targets.
- `target_arranger_state.npy`: `[N, 8, T]` explicit bass-rhythm, bass-sustain, chord-body, inner-motion, top-line, section-bass, section-fullness, and section-transition state.
- `target_bass_continuity.npy`: `[N, 4, T]` split bass-rhythm, bass-sustain, section-bass, and section-transition state.
- `target_body_melody_state.npy`: `[N, 6, T]` split chord-body, inner-motion, top-line, high-activity, section-body, and section-melody state.
- `target_section_diversity.npy`: `[N, 4, T]` section pitch-variety, pitch-class, range, and onset-density targets.
- `index.csv`: source metadata and note counts.
- `meta.json`: shape, feature, and config metadata.

## Smoke Training

The first model is a small temporal Conv1D generator with separate onset, frame, velocity, and pedal heads. It is not the final architecture; it proves that the cache can train a new model and round-trip predictions into arrangement artifacts.

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\smoke_2track" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "smoke_train_2track" `
  --epochs 2 `
  --batch-size 2 `
  --hidden-channels 32 `
  --n-blocks 2 `
  --sample-count 1 `
  --min-selected-notes 24 `
  --device cpu
```

## Model Inference

After a smoke or full run, render an arbitrary source clip through the checkpoint:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\smoke_train_2track_midifix\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --max-frames 128 `
  --out-stem "saves2\piano_arranger\outputs\model_smoke\song_model_piano" `
  --max-notes-per-second 32 `
  --max-simultaneous-notes 12 `
  --max-onsets-per-frame 6 `
  --max-pitch-fraction 0.22 `
  --max-pitch-class-fraction 0.32 `
  --max-note-duration 1.5 `
  --min-selected-notes 24 `
  --min-unique-pitches 8 `
  --device cpu
```

This writes MIDI, JSON, summary, and optional preview WAV. The current smoke checkpoint is only proof that the model path runs; quality work still depends on scaling data and replacing heuristic pseudo-targets.

## Structural Evaluation

Run the automated gate on any arrangement JSON:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action evaluate `
  --arrangement-json "saves2\piano_arranger\outputs\model_smoke\moonlight_4s_model.json" `
  --eval-report "saves2\piano_arranger\reports\moonlight_4s_model_eval.json" `
  --eval-label model_smoke
```

The current heuristic smoke sample passes this gate. The current tiny trained smoke model fails it for `overdense` and `overstacked`, with roughly 131 notes/second and 94 simultaneous notes. Future training work should make model samples pass these structural checks before manual listening claims are trusted.

The same tiny checkpoint can now pass the structural gate when decoded with playable constraints:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\smoke_train_2track_midifix\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --max-frames 128 `
  --out-stem "saves2\piano_arranger\outputs\model_smoke_decodefix\song_model_piano_decodefix" `
  --max-notes-per-second 32 `
  --max-simultaneous-notes 12 `
  --max-onsets-per-frame 6 `
  --max-pitch-fraction 0.22 `
  --max-pitch-class-fraction 0.32 `
  --min-selected-notes 24 `
  --device cpu
```

Verified smoke evidence: `moonlight_4s_model_decodefix_v3` produced 51 notes, passed all structural warnings, parsed as MIDI with 51 `note_on` events, and rendered a valid 22.05 kHz preview WAV. This is a decode validity gate, not final model quality.

Source-aware evaluation compares arrangement JSON against the original audio:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action evaluate `
  --arrangement-json "saves2\piano_arranger\outputs\model_smoke_decodefix\moonlight_4s_model_decodefix_v3.json" `
  --eval-report "saves2\piano_arranger\reports\moonlight_4s_model_decodefix_v3_source_eval.json" `
  --eval-label model_smoke_decodefix_v3_source `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --frame-hz 25 `
  --max-frames 128
```

Current source-aware evidence: the heuristic baseline passes with strong source alignment (`global chroma=0.929`, `active chroma=0.777`, `onset corr=0.331`). The constrained smoke model is playable but still fails active harmony alignment (`active chroma=0.149`), so the next training work must improve source-following rather than only pruning decode output.

## Larger Bootstrap Smoke

The first scaled smoke cache/run is:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action cache `
  --max-tracks 8 `
  --cache-seconds 4 `
  --max-frames 128 `
  --cache-dir "saves2\piano_arranger\cache\bootstrap_8track_4s"

python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\bootstrap_8track_4s" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "bootstrap_8track_4s_eval_smoke" `
  --epochs 2 `
  --batch-size 2 `
  --hidden-channels 48 `
  --n-blocks 3 `
  --model-architecture key_conditioned `
  --key-embed-dim 32 `
  --sample-count 2 `
  --density-loss-weight 0.35 `
  --chroma-loss-weight 0.35 `
  --pitch-usage-loss-weight 0.75 `
  --min-selected-notes 24 `
  --device cpu
```

Evidence: the 8-track cache built with 0 errors and the 2-epoch baseline run reduced loss from `2.4355` to `2.2087`. Epoch sample evaluation still failed: outputs were too sparse and pitch-collapsed. After fixing overaggressive dominance pruning, `moonlight_4s_threshold020_prunefix` produced valid MIDI/WAV with 15 notes, but still failed pitch/drone/harmony/register collapse gates.

Auxiliary supervised losses are available:

- `--density-loss-weight`: compares predicted and target onset/frame activity curves.
- `--chroma-loss-weight`: compares frame-level pitch-class distributions.
- `--pitch-usage-loss-weight`: compares normalized 88-key usage over the clip.
- `--hierarchy-loss-weight`: trains density/register planning heads when hierarchy targets are available.
- `--musical-plan-loss-weight`: trains chord, bass/root-proxy, and voicing planning heads when musical-plan targets are available.
- `--event-plan-loss-weight`: trains onset/off/change/chord-change event planning heads.
- `--pc-onset-plan-loss-weight`: trains pitch-class onset planning heads.
- `--pc-onset-f1-loss-weight`: trains a soft +/-1-frame pitch-class onset F1 surrogate aligned with the strict target pitch-class-onset validator.
- `--pc-onset-alignment-loss-weight`: trains the actual note onset heads by aggregating predicted key onsets into pitch classes and matching `target_pc_onset`.
- `--role-plan-loss-weight`: trains bass/chord/melody/polyphony/velocity role planning heads for learned fullness.
- `--texture-balance-loss-weight`: directly trains note probabilities toward target register balance, mid-register body, and frame density.
- `--melody-plan-loss-weight`: trains explicit high/upper/top-pitch/top-velocity melody planning heads.
- `--melody-balance-loss-weight`: directly pressures note probabilities toward the target top-line register and pitch contour.
- `--texture-role-plan-loss-weight`: trains the joint bass/body/inner/top-line texture-role planning head.
- `--texture-role-balance-loss-weight`: directly pressures note probabilities to preserve those roles together.
- `--section-role-plan-loss-weight`: trains the section-level bass/body/melody/fullness continuity planning head.
- `--section-role-balance-loss-weight`: directly pressures note probabilities to match section-level role coverage, with extra penalty for under-predicted bass coverage.
- `--arranger-state-plan-loss-weight`: trains the explicit 8-channel section/role arranger state injected into per-key note heads.
- `--bass-continuity-plan-loss-weight`: trains the split left-hand continuity state.
- `--body-melody-state-plan-loss-weight`: trains the split chord-body/top-line state.
- `--body-melody-state-balance-loss-weight`: directly pressures note probabilities to preserve split chord-body/top-line coverage.
- `--section-diversity-plan-loss-weight`: trains the section-level unique-pitch, pitch-class, range, and onset-density planning head.
- `--section-diversity-balance-loss-weight`: directly pressures note probabilities to avoid falling below section pitch variety, range, and onset-density targets. This term is one-sided underuse pressure; the earlier symmetric version was negative evidence because it could reward thin output when sparse pseudo-target sections had low density.
- `--section-diversity-guidance-weight` with `--section-diversity-reserve-fraction`: uses the learned section-diversity head during decode to reserve a bounded number of section-local pitch/pitch-class/range candidates. Keep this opt-in while gathering evidence; it is learned-head guidance, not final proof of arrangement quality.
- `--anti-collapse-loss-weight`: penalizes low pitch and pitch-class entropy in soft frame probabilities.
- `--warm-start-checkpoint`: initializes a train run from a previous checkpoint by loading only matching tensor names/shapes. Use this to keep the strong role/texture body while adding newer section-diversity or arranger-state heads.
- `--sample-score-min-mid-note-fraction` and `--sample-score-max-high-note-fraction`: add sample-selector penalties for mid-register underuse and high-register dominance. Use these on warm-start runs so loud/high-heavy samples do not win over balanced piano body.

Sample checkpoint scoring is role-aware. Training sample eval now aggregates chord-frame coverage, bass coverage, melody coverage, mean active polyphony, fullness score, and rendered WAV RMS, then writes `sample_eval_score_components` into `history.json`. Tune it with `--sample-score-role-balance-weight`, `--sample-score-chord-frame-target`, `--sample-score-melody-coverage-target`, `--sample-score-bass-coverage-min`, `--sample-score-bass-coverage-max`, `--sample-score-polyphony-target`, and `--sample-score-rms-target`. It also has joint quality penalties for thin density, low RMS, pitch-class dominance, and weak section minima; tune those with `--sample-score-quality-penalty-weight`, `--sample-score-min-notes-per-second`, `--sample-score-min-section-notes`, `--sample-score-min-section-unique-pitches`, `--sample-score-min-section-chord-frame`, `--sample-score-min-section-fullness`, and `--sample-score-max-single-pitch-class-fraction`. Keep pass/warning gates active; the score is a checkpoint selector, not a license to accept collapsed samples.

Sample checkpoint scoring is also section-aware. Training sample eval runs the section reporter by default, appends section warnings to the sample warnings, and records section minima in history: `sample_eval_mean_min_section_notes`, `sample_eval_mean_min_section_unique_pitches`, `sample_eval_mean_min_section_bass_coverage_fraction`, `sample_eval_mean_min_section_chord_frame_fraction`, and `sample_eval_mean_min_section_fullness_score`. Use `--no-section-sample-eval` only for speed/debug, and `--sample-section-seconds` to change the section size.

Chunked decode has an optional section bass repair guard. Use `--section-bass-repair` with `--section-bass-repair-min-coverage` when a globally valid artifact has isolated bassless sections; the repair inserts a protected low-register note only when the local register-coverage floor is missed and still obeys pitch/polyphony caps. Treat it as a continuity guard for inference, not proof that the model has learned section-level bass planning.

Comparison evidence on the same 8-track cache:

- `bootstrap_8track_4s_eval_smoke`: mean notes/sec `0.879`, source active chroma `0.258`, onset corr `0.005`.
- `bootstrap_8track_4s_auxloss_smoke`: mean notes/sec `2.539`, source active chroma `0.224`, onset corr `0.104`.
- `bootstrap_8track_4s_pitchloss_scaled_smoke`: mean notes/sec `2.441`, source active chroma `0.335`, onset corr `0.117`, but still pitch-collapsed.
- `bootstrap_8track_4s_keyconditioned_smoke`: mean notes/sec `3.779`, source active chroma `0.284`, onset corr `0.146`; raw inference loaded the keyed checkpoint and rendered valid MIDI/WAV, but still pitch-collapsed with 3 unique pitches on the checked Moonlight output.

The current model architectures are:

- `conv1d`: temporal backbone with direct 88-key heads.
- `key_conditioned`: temporal backbone plus learned key embeddings for per-key onset/frame/velocity heads.
- `chroma_key_conditioned`: key-conditioned heads plus source-chroma injection for each piano key's pitch class. This is the first architecture aimed directly at the paired path's active-harmony failure.
- `harmony_conditioned`: chroma-key style conditioning plus a supervised 12-channel harmonic-plan head injected into per-key onset/frame/velocity heads. Train its plan head with `--harmonic-plan-loss-weight`.
- `musical_plan_conditioned`: harmony-conditioned structure plus predicted chord, bass/root-proxy, voicing, event, pitch-class-onset, role/fullness, melody/top-line, joint texture-role, section-role, explicit arranger-state, split bass-continuity/body-melody, and section-diversity plans injected into per-key onset/frame/velocity heads. This is the first architecture where the new musical-plan representation changes note generation directly.

All architectures also emit shared musical-plan heads: `chord_logits`, `bass_logits`, and `voicing`. The cache writes `target_chord`, `target_bass`, and `target_voicing`; older caches derive those targets on load. Train the heads with `--musical-plan-loss-weight`. Decode can optionally use those plans with `--chord-plan-guidance-weight`, `--bass-plan-guidance-weight`, and `--voicing-plan-guidance-weight`.

All architectures also emit a pitch-class-onset plan: `pc_onset_logits` / `pc_onset`. The cache writes `target_pc_onset (N,12,T)` and older caches derive it from `target_onset` on load. Train this direct event-fidelity head with `--pc-onset-plan-loss-weight`; add `--pc-onset-f1-loss-weight` when the goal is to optimize the strict target pitch-class-onset F1 gap directly; use `--pc-onset-alignment-loss-weight` to apply pitch-class onset pressure to the actual note onset heads. Decode can optionally use the plan with `--pc-onset-plan-guidance-weight`.

Pc-onset plan reserve decode can create extra candidates from high-confidence local maxima in the predicted pitch-class onset plan before the per-frame candidate cap. Use `--pc-onset-plan-reserve-threshold`, `--pc-onset-plan-reserve-max-per-frame`, and `--pc-onset-plan-reserve-min-note-score`. This is inference-compatible and metadata records `decode_pc_onset_reserved_candidates`; it is a pressure-test mechanism for event selection, not a quality guarantee.

Paired validation now includes raw pc-onset plan diagnostics when target eval is enabled. The validator runs the checkpoint on each source and sweeps local maxima in `pred["pc_onset"]` against the target MIDI pitch-class onsets, writing `pc_onset_plan_diagnostics` per row plus scalar summary fields such as `pc_onset_plan_best_f1`, threshold, precision, recall, predicted count, and target count. Use this to separate "the model lacks the event signal" from "decode failed to select the event signal."

All architectures also emit a role/fullness plan: `role (N,5,T)`. The cache writes `target_role` for bass activity, chord-frame activity, melody-register activity, normalized polyphony, and active-frame velocity weight; older caches derive it from target frame/velocity rolls. Train this direct fullness head with `--role-plan-loss-weight`.

All architectures also emit a melody/top-line plan: `melody (N,4,T)`. The cache writes `target_melody` for high-register activity, upper-register activity, normalized top pitch, and top velocity; older caches derive it from target frame/velocity rolls. Train this direct top-line head with `--melody-plan-loss-weight`, and use `--melody-balance-loss-weight` to pressure note probabilities toward the target top-line contour.

All architectures also emit a joint texture-role plan: `texture_role (N,4,T)`. The cache writes `target_texture_role` for bass floor, chord body, inner motion, and top-line presence; older caches derive it from target onset/frame rolls. Train this direct joint-role head with `--texture-role-plan-loss-weight`, and use `--texture-role-balance-loss-weight` to preserve those arrangement roles together in note probabilities.

All architectures also emit a section-role plan: `section_role (N,4,T)`. The cache writes `target_section_role` for local section bass coverage, chord body, melody presence, and fullness broadcast across each section; older caches derive it from target onset/frame rolls. Train this continuity head with `--section-role-plan-loss-weight`, and use `--section-role-balance-loss-weight` when the goal is for the model itself to maintain section-level bass/body/melody coverage instead of relying on decode repair.

All architectures also emit an explicit arranger-state plan: `arranger_state (N,8,T)`. The cache writes `target_arranger_state` for bass rhythm, bass sustain, chord body, inner motion, top line, section bass continuity, section fullness, and section-transition emphasis; older caches derive it from target onset/frame rolls. `musical_plan_conditioned` injects this state into per-key note heads with a zero-initialized projection. Train it with `--arranger-state-plan-loss-weight`; there is intentionally no direct balance loss yet.

All architectures also emit split arranger-state plans: `bass_continuity (N,4,T)` and `body_melody_state (N,6,T)`. The cache writes `target_bass_continuity` for bass rhythm, bass sustain, section bass continuity, and section transitions, plus `target_body_melody_state` for chord body, inner motion, top line, high activity, section body, and section melody; older caches derive both on load. `musical_plan_conditioned` injects both with zero-initialized projections. Train them with `--bass-continuity-plan-loss-weight` and `--body-melody-state-plan-loss-weight`; use `--body-melody-state-balance-loss-weight` only when directly testing note-probability pressure for body/top-line coverage.

All architectures also emit a section-diversity plan: `section_diversity (N,4,T)`. The cache writes `target_section_diversity` for local section unique-pitch coverage, pitch-class coverage, pitch range, and onset density; older caches derive it from target onset/frame rolls. Train this head with `--section-diversity-plan-loss-weight`, and use `--section-diversity-balance-loss-weight` to counter the narrow-pitch section-role failure mode.

The next model work should improve target representation and pitch-structured decoding for diversity and onset timing, not just tune decode constraints or loss weights. The current pressure point is chord/bass/root-proxy/voicing and event-level target fidelity: coarse chroma can pass while pitch-class-onset F1 is still low.

Structured diversity decoding is available for inspectable samples:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\bootstrap_8track_4s_keyconditioned_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --max-frames 128 `
  --out-stem "saves2\piano_arranger\outputs\bootstrap_8track_4s_keyconditioned_smoke\song_structured" `
  --model-architecture key_conditioned `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 16 `
  --min-unique-pitches 8 `
  --diversity-fallback-threshold 0.0 `
  --device cpu
```

Evidence: `moonlight_4s_keyconditioned_structured_protected` passed the current automated gates with 16 notes, 8 unique pitches, low/mid/high coverage, source active chroma `0.432`, and onset corr `0.232`; MIDI and WAV both validated. This is still a scaffolded short sample, not the final full-arrangement target.

Hierarchy target smoke:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action cache `
  --max-tracks 4 `
  --cache-seconds 4 `
  --max-frames 128 `
  --cache-dir "saves2\piano_arranger\cache\hierarchy_4track_4s_smoke"

python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\hierarchy_4track_4s_smoke" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "hierarchy_4track_4s_keyconditioned_smoke" `
  --epochs 2 `
  --batch-size 2 `
  --model-architecture key_conditioned `
  --hierarchy-loss-weight 0.25 `
  --sample-count 2 `
  --device cpu
```

Evidence: the hierarchy cache wrote `[2,128]` density and `[3,128]` register targets with 0 errors. The train smoke logged `hierarchy_loss` decreasing from `0.840` to `0.716` and validated sample MIDI/WAV. The remaining failure is source rhythm/onset alignment, not cache shape or renderer plumbing.

Source-onset guided decoding is available for rhythm alignment:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\hierarchy_4track_4s_keyconditioned_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --out-stem "saves2\piano_arranger\outputs\hierarchy_4track_4s_keyconditioned_smoke\song_rhythm_guided" `
  --onset-threshold 0.35 `
  --frame-threshold 0.35 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 16 `
  --min-unique-pitches 8 `
  --diversity-fallback-threshold 0.0 `
  --source-onset-guidance-weight 1.0 `
  --source-onset-snap-frames 2 `
  --source-onset-peak-threshold 0.35 `
  --device cpu
```

Comparison evidence on the same hierarchy checkpoint and Moonlight 4-second source: unguided decode failed source-aware evaluation (`source_onset_correlation=-0.147`, `source_onset_peak_alignment=0.000`, `source_active_chroma_cosine=0.089`), while guided decode passed with 19 notes, 8 unique pitches, `source_onset_correlation=0.460`, `source_onset_peak_alignment=0.667`, `source_active_chroma_cosine=0.345`, and valid MIDI/WAV. This is a rhythm-planning scaffold for inspectable samples; the model still needs to learn the alignment rather than rely on decode snapping.

Trained source-onset planning is available:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\hierarchy_4track_4s_smoke" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "hierarchy_4track_4s_sourceonset20_smoke" `
  --epochs 20 `
  --batch-size 2 `
  --model-architecture key_conditioned `
  --hierarchy-loss-weight 0.25 `
  --source-onset-loss-weight 2.0 `
  --sample-every 5 `
  --sample-count 2 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --device cpu
```

Render from that checkpoint with direct source-onset snapping disabled but learned density-plan timing and dynamic velocity enabled:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\hierarchy_4track_4s_sourceonset20_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --out-stem "saves2\piano_arranger\outputs\hierarchy_4track_4s_sourceonset20_smoke\song_densityplan_dynamic" `
  --onset-threshold 0.35 `
  --frame-threshold 0.35 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 16 `
  --min-unique-pitches 8 `
  --diversity-fallback-threshold 0.0 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --source-energy-velocity-weight 1.0 `
  --density-plan-velocity-weight 0.75 `
  --device cpu
```

Evidence: `source_onset_loss` decreased from `0.503` to `0.144` in the 20-epoch smoke run. `moonlight_4s_densityplan_dynamic` passed source-aware evaluation with no warnings: 28 notes, 9 unique pitches, `source_onset_correlation=0.272`, `source_active_chroma_cosine=0.259`, `velocity_range=58`, `velocity_std=15.739`; MIDI and WAV loaded successfully. Metadata showed direct source-onset snapping was off (`snapped_candidates=0`) and learned density-plan snapping was active (`snapped_candidates=334`). This is still pseudo-target smoke evidence, not final arrangement quality.

The same trained-planning setup has been scaled to the 8-track bootstrap cache:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\bootstrap_8track_4s" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "bootstrap_8track_4s_sourceonset20_smoke" `
  --epochs 20 `
  --batch-size 2 `
  --model-architecture key_conditioned `
  --hierarchy-loss-weight 0.25 `
  --source-onset-loss-weight 2.0 `
  --sample-every 5 `
  --sample-count 3 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --source-energy-velocity-weight 1.0 `
  --density-plan-velocity-weight 0.75 `
  --device cpu
```

Render/evaluate the current best smoke checkpoint:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\bootstrap_8track_4s_sourceonset20_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --out-stem "saves2\piano_arranger\outputs\bootstrap_8track_4s_sourceonset20_smoke\song_densityplan_dynamic" `
  --onset-threshold 0.35 `
  --frame-threshold 0.35 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 16 `
  --min-unique-pitches 8 `
  --diversity-fallback-threshold 0.0 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --source-energy-velocity-weight 1.0 `
  --density-plan-velocity-weight 0.75 `
  --device cpu
```

Evidence: `bootstrap_8track_4s_sourceonset20_smoke` reduced `source_onset_loss` from `0.444` to `0.098`. Epoch 15 passed all 3 source-aware samples with mean onset correlation `0.543`; epoch 20 passed 2/3 with mean onset correlation `0.545` and one `register_underuse` warning. `moonlight_4s_densityplan_dynamic` from the epoch-20 checkpoint passed source-aware evaluation with no warnings: 24 notes, 10 unique pitches, `source_onset_correlation=0.594`, `source_onset_peak_alignment=0.750`, `source_active_chroma_cosine=0.278`, `velocity_range=62`, `velocity_std=15.659`; MIDI and WAV loaded successfully. Metadata showed direct source-onset snapping was off (`snapped_candidates=0`) and learned density-plan snapping was active (`snapped_candidates=1082`). This is the best automated smoke result so far, but it is still pseudo-target and 4-second evidence only.

Longer-clip smoke:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\bootstrap_8track_4s_sourceonset20_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 12 `
  --max-frames 300 `
  --frame-hz 25 `
  --out-stem "saves2\piano_arranger\outputs\bootstrap_8track_4s_sourceonset20_smoke\song_12s_densityplan_dynamic_chunkreg" `
  --onset-threshold 0.35 `
  --frame-threshold 0.35 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 48 `
  --min-unique-pitches 12 `
  --register-coverage-chunk-seconds 4 `
  --diversity-fallback-threshold 0.0 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --source-energy-velocity-weight 1.0 `
  --density-plan-velocity-weight 0.75 `
  --device cpu
```

Evidence: a first 12-second run without chunk register reservation had strong rhythm (`source_onset_correlation=0.691`, `source_onset_peak_alignment=0.952`) but failed `register_underuse` because high-register usage was `0.029`. With `--register-coverage-chunk-seconds 4`, `moonlight_12s_densityplan_dynamic_chunkreg` passed with no warnings: 65 notes, 13 unique pitches, high-register fraction `0.062`, `source_onset_correlation=0.677`, `source_onset_peak_alignment=0.905`, `source_active_chroma_cosine=0.311`, `velocity_range=73`, `velocity_std=14.028`; MIDI and WAV loaded successfully. Metadata showed direct source-onset snapping was off, learned density-plan snapping was active (`3549` snaps), and `6` chunk register reservations were used. This is 12-second smoke evidence, not full-song proof.

30-second smoke and section reporting:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer `
  --checkpoint "saves2\piano_arranger\runs\bootstrap_8track_4s_sourceonset20_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 30 `
  --max-frames 750 `
  --frame-hz 25 `
  --out-stem "saves2\piano_arranger\outputs\bootstrap_8track_4s_sourceonset20_smoke\song_30s_densityplan_dynamic_chunkreg" `
  --onset-threshold 0.35 `
  --frame-threshold 0.35 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 120 `
  --min-unique-pitches 16 `
  --register-coverage-chunk-seconds 4 `
  --diversity-fallback-threshold 0.0 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --source-energy-velocity-weight 1.0 `
  --density-plan-velocity-weight 0.75 `
  --device cpu

python "lab 3.6\run_piano_arranger_pipeline.py" --action section-report `
  --arrangement-json "saves2\piano_arranger\outputs\bootstrap_8track_4s_sourceonset20_smoke\song_30s_densityplan_dynamic_chunkreg.json" `
  --eval-report "saves2\piano_arranger\reports\song_30s_densityplan_dynamic_chunkreg_sections.json" `
  --section-seconds 5
```

Evidence: `moonlight_30s_densityplan_dynamic_chunkreg` passed source-aware evaluation with no warnings: 142 notes, 18 unique pitches, high-register fraction `0.077`, `source_onset_correlation=0.681`, `source_onset_peak_alignment=0.959`, `source_active_chroma_cosine=0.273`, `velocity_range=74`, `velocity_std=13.271`; MIDI and WAV loaded successfully. Metadata showed direct source-onset snapping was off, learned density-plan snapping was active (`8603` snaps), and `21` chunk register reservations were used. The 5-second section report had 6 sections, no empty sections, no high-register-absent sections, min section notes `22`, max section notes `38`, and min section high-register fraction `0.032`. Residual issue: section-level pitch diversity can still sag; the last 5-second section had only 6 unique pitches.

Chunked inference is available for longer sources:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action infer-chunked `
  --checkpoint "saves2\piano_arranger\runs\bootstrap_8track_4s_sourceonset20_smoke\checkpoints\latest.pt" `
  --source-audio "<song.wav>" `
  --seconds 30 `
  --max-frames 300 `
  --frame-hz 25 `
  --chunk-seconds 12 `
  --chunk-hop-seconds 12 `
  --section-profile flat `
  --out-stem "saves2\piano_arranger\outputs\bootstrap_8track_4s_sourceonset20_smoke\song_30s_chunked_12s_densityplan_dynamic_chunkreg" `
  --onset-threshold 0.35 `
  --frame-threshold 0.35 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 120 `
  --min-unique-pitches 16 `
  --register-coverage-chunk-seconds 4 `
  --diversity-fallback-threshold 0.0 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 2 `
  --density-plan-peak-threshold 0.35 `
  --source-energy-velocity-weight 1.0 `
  --density-plan-velocity-weight 0.75 `
  --device cpu
```

`--section-profile flat` keeps chunk density, diversity, velocity, and register controls constant. `--section-profile arc` makes edge chunks lighter and the middle chunk denser/louder; the applied multipliers are stored in per-chunk metadata. This is deterministic section-control scaffolding, not learned musical form.

Evidence: `moonlight_30s_chunked_12s_densityplan_dynamic_chunkreg` passed source-aware evaluation with no warnings: 184 notes, 15 unique pitches, high-register fraction `0.076`, `source_onset_correlation=0.722`, `source_onset_peak_alignment=0.841`, `source_active_chroma_cosine=0.334`, `velocity_range=73`, `velocity_std=14.728`; MIDI and WAV loaded successfully. The stricter 5-second section report v2 had 6 nonempty sections, no high-register-absent sections, min section notes `24`, max section notes `46`, min section high-register fraction `0.026`, and no `section_pitch_underuse` warnings.

Arc evidence: `moonlight_30s_chunked_12s_arc_densityplan_dynamic_chunkreg` passed aggregate source-aware evaluation with no warnings: 180 notes, 14 unique pitches, high-register fraction `0.078`, `source_onset_correlation=0.702`, `source_onset_peak_alignment=0.850`, `source_active_chroma_cosine=0.310`, `velocity_range=70`, `velocity_std=15.688`; MIDI and WAV loaded successfully. Its section report v2 warned `section_pitch_underuse:1` because section 1 had only 7 unique pitches. Flat chunking is currently cleaner by automated section diversity; arc is useful because it demonstrates form-control hooks and exposes the next learned-planning problem.

Section reports warn on `empty_section:<idx>`, `section_high_register_absent:<idx>`, and `section_pitch_underuse:<idx>` when a section has at least 8 notes but fewer than 8 unique pitches. The combined arrangement metadata stores per-chunk decode metadata. This is infrastructure for full-song inference, not proof of full-song musical form.

## MIDI Target Path

Real symbolic targets can now be discovered and cached. This is the replacement path for heuristic pseudo-targets once a real piano MIDI or transcription corpus is available.

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action discover-midi `
  --midi-root "<folder-with-midi-files>" `
  --midi-manifest "data\piano_arranger_manifests\midi_piano_targets.csv" `
  --report-path "data\piano_arranger_manifests\midi_piano_targets.summary.json" `
  --min-midi-notes 8

python "lab 3.6\run_piano_arranger_pipeline.py" --action midi-cache `
  --midi-manifest "data\piano_arranger_manifests\midi_piano_targets.csv" `
  --cache-dir "saves2\piano_arranger\cache\midi_targets_v1" `
  --cache-seconds 8 `
  --max-frames 256 `
  --frame-hz 25 `
  --min-midi-notes 8 `
  --midi-source-preview-mode ensemble
```

MIDI discovery excludes package examples, virtualenvs, and source-control folders by default. Use `--include-package-midi-examples` only for parser smoke tests, never as training evidence.

`--midi-source-preview-mode piano` preserves the original deterministic piano-preview conditioning. `--midi-source-preview-mode ensemble` renders a deterministic non-piano synthetic source from the same MIDI while keeping the symbolic piano-roll as the target. Use `ensemble` for the next real-corpus experiments because it is closer to song-to-piano arrangement than training the model to reconstruct piano from piano.

Current evidence:

- Real local scan: `Z:\DataSets` had `discovered_files=0`, `selected_rows=0`, `errors=0`; see `data\piano_arranger_manifests\midi_piano_targets.summary.json`.
- Parser-only smoke manifest: `data\piano_arranger_manifests\midi_parser_smoke.csv` from `.venv_lab1\Lib\site-packages\miditoolkit\midi\examples_data\1390.mid`.
- Parser-only smoke cache: `saves2\piano_arranger\cache\midi_parser_smoke`, `target_source=midi_symbolic_target_v1`, 1 sample, 0 errors, nonzero onset/frame/density/register targets.
- Parser-only train smoke: `saves2\piano_arranger\runs\midi_parser_smoke_train`, 1 CPU epoch, checkpoint written, sample structural eval passed with 50 notes and 8 unique pitches; MIDI and WAV loaded successfully.
- Parser-only ensemble cache: `saves2\piano_arranger\cache\midi_parser_smoke_ensemble`, 1 sample, 0 errors, `conditioning_source=deterministic_ensemble_preview_audio_from_midi_target`, source shape `[17,128]`, target shapes `[88,128]`, and hierarchy target shapes `[2,128]` plus `[3,128]`.
- Parser-only ensemble train smoke: `saves2\piano_arranger\runs\midi_parser_smoke_ensemble_train`, 1 CPU epoch, checkpoint written, sample structural eval passed with 24 notes, 9 unique pitches, max simultaneity 9, velocity range 26; sample MIDI parsed with 24 `note_on` events and WAV loaded at 22.05 kHz. Source-aware sample eval is skipped for MIDI rows because the manifest path is symbolic, not audio.

This proves the real-target cache and training path works, including the stronger synthetic source-to-piano variant. It does not prove musical quality because the smoke input is a package example, not a curated corpus.

## Paired Source/Target Path

Use this path when a source song audio file has a corresponding piano arrangement or transcription MIDI. This is the closest current training contract to the real goal: source audio in, symbolic piano performance out.

Paired manifest contract:

```csv
source_audio,target_midi,title,artist,source
<song.wav>,<piano_target.mid>,<title>,<artist>,<provenance>
```

Build the cache:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action discover-paired `
  --audio-root "<folder-with-source-audio>" `
  --midi-root "<folder-with-target-midi>" `
  --paired-manifest "data\piano_arranger_manifests\paired_audio_midi_targets.csv" `
  --report-path "data\piano_arranger_manifests\paired_audio_midi_targets.summary.json" `
  --min-midi-notes 8

python "lab 3.6\run_piano_arranger_pipeline.py" --action review-paired `
  --paired-manifest "data\piano_arranger_manifests\paired_audio_midi_targets.csv" `
  --report-path "data\piano_arranger_manifests\paired_audio_midi_targets.audit.json" `
  --min-midi-notes 8 `
  --max-pair-duration-delta 5

python "lab 3.6\run_piano_arranger_pipeline.py" --action paired-cache `
  --paired-manifest "data\piano_arranger_manifests\paired_audio_midi_targets.audit.passed.csv" `
  --cache-dir "saves2\piano_arranger\cache\paired_audio_midi_v1" `
  --cache-seconds 8 `
  --max-frames 256 `
  --frame-hz 25 `
  --min-midi-notes 8
```

`discover-paired` matches files by normalized stem. Exact matches are strongest; prefix matches are useful for generated names like `song_transcription.mid` or `song_arranged.mid`, but they need manual review before real training.

`review-paired` probes the source audio and target MIDI before cache building. It writes the JSON report, a `.rows.csv` file, and a `.passed.csv` manifest containing only warning-free rows. It flags missing files, prefix matches, weak MIDI targets, probe errors, and duration mismatches. Use `.passed.csv` for `paired-cache` unless every warning has been deliberately accepted.

Current parser-only evidence:

- Smoke manifest: `data\piano_arranger_manifests\paired_parser_smoke.csv`, with generated source WAV `saves2\piano_arranger\outputs\paired_parser_smoke\1390_ensemble_source_4s.wav` and package MIDI target `.venv_lab1\Lib\site-packages\miditoolkit\midi\examples_data\1390.mid`.
- Smoke cache: `saves2\piano_arranger\cache\paired_parser_smoke`, 1 sample, 0 errors, `target_source=paired_audio_midi_symbolic_target_v1`, `conditioning_source=manifest_source_audio`, source shape `[17,128]`, target shapes `[88,128]`, and hierarchy target shapes `[2,128]` plus `[3,128]`.
- Smoke train: `saves2\piano_arranger\runs\paired_parser_smoke_train`, 1 CPU epoch, checkpoint written, sample MIDI parsed with 32 `note_on` events and WAV loaded at 22.05 kHz.
- The one-epoch sample was structurally valid but failed source-aware metrics: global chroma `0.133`, active chroma `0.076`, onset correlation `-0.062`, with source harmony/rhythm warnings. That failure is useful evidence: this path exercises the real source-following problem rather than structural validity alone.
- Discovery smoke: `data\piano_arranger_manifests\paired_parser_discovered.csv` and `.summary.json` were generated by scanning the parser smoke source WAV folder and package MIDI examples with explicit `--include-package-midi-examples`. It found 1 audio file, 1 MIDI file, selected 1 prefix-stem pair, and had 0 errors.
- Review smoke: `data\piano_arranger_manifests\paired_parser_discovered.audit.json`, `.audit.rows.csv`, and `.audit.passed.csv` flagged the discovered parser pair with `prefix_match_needs_review` and `duration_mismatch`; source audio duration was 4.82s and target MIDI duration was 78.99s. This is correct for the smoke and proves the audit catches suspicious pairs. The passed manifest contains only headers.
- Discovery cache smoke: `saves2\piano_arranger\cache\paired_parser_discovered`, built from the discovered paired manifest with 1 sample, 0 errors, source shape `[17,128]`, target shapes `[88,128]`, and hierarchy target shapes `[2,128]` plus `[3,128]`.
- Positive passed-manifest smoke: `saves2\piano_arranger\outputs\paired_parser_exact_smoke\1390.wav` and cropped target `1390.mid` were discovered as an exact-stem pair. `data\piano_arranger_manifests\paired_parser_exact_discovered.audit.json` passed 1/1 rows with no warnings and wrote `data\piano_arranger_manifests\paired_parser_exact_discovered.audit.passed.csv`.
- Passed-manifest cache smoke: `saves2\piano_arranger\cache\paired_parser_exact_passed`, built from `.audit.passed.csv`, 1 sample, 0 errors, source shape `[17,128]`, target shapes `[88,128]`, and hierarchy target shapes `[2,128]` plus `[3,128]`.

For real work, prefer curated paired rows over parser smoke. Use `paired-cache` when true pairs exist; use `midi-cache --midi-source-preview-mode ensemble` only as a pre-paired bootstrap.

Paired overfit smoke:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\paired_parser_exact_passed" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "paired_parser_exact_overfit20" `
  --epochs 20 `
  --batch-size 1 `
  --hidden-channels 48 `
  --n-blocks 3 `
  --model-architecture key_conditioned `
  --key-embed-dim 24 `
  --hierarchy-loss-weight 0.25 `
  --source-onset-loss-weight 0.5 `
  --sample-every 5 `
  --sample-count 1 `
  --min-selected-notes 24 `
  --min-unique-pitches 8 `
  --device cpu
```

Evidence: `paired_parser_exact_overfit20` reduced loss from `4.716` to `3.557` and source-onset loss from `0.611` to `0.196`. Epoch samples improved source-onset correlation from `0.237` at epoch 5 to `0.522` at epoch 20, but still failed automated sample eval at epoch 20 due to `register_underuse` and `source_active_harmony_mismatch`.

Standalone inference from the epoch-20 checkpoint rendered `saves2\piano_arranger\outputs\paired_parser_exact_overfit20\1390_overfit20_densityplan.*`; MIDI parsed with 47 `note_on` events and WAV loaded at 22.05 kHz. Source-aware eval report `saves2\piano_arranger\reports\paired_parser_exact_overfit20_densityplan_source_eval.json` failed only `source_active_harmony_mismatch`: 47 notes, 13 unique pitches, velocity range `59`, source global chroma `0.307`, active chroma `0.183`, onset correlation `0.552`, peak alignment `0.688`.

Interpretation: the reviewed paired path can train source-rhythm behavior, but harmony following is still weak. The next model work should improve harmonic conditioning/targets rather than adding more decode patches.

Source-chroma alignment loss is available:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\paired_parser_exact_passed" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "paired_parser_exact_overfit20_sourcechroma" `
  --epochs 20 `
  --batch-size 1 `
  --model-architecture key_conditioned `
  --source-onset-loss-weight 0.5 `
  --source-chroma-loss-weight 1.0 `
  --sample-every 5 `
  --sample-count 1 `
  --device cpu
```

Evidence: `paired_parser_exact_overfit20_sourcechroma` reduced loss from `4.799` to `3.640`, but `source_chroma_loss` only moved from `0.0835` to `0.0829`. Standalone inference rendered `saves2\piano_arranger\outputs\paired_parser_exact_overfit20_sourcechroma\1390_overfit20_sourcechroma_densityplan.*`; MIDI/WAV loaded and the output had 47 notes. Source-aware eval `saves2\piano_arranger\reports\paired_parser_exact_overfit20_sourcechroma_densityplan_source_eval.json` still failed only `source_active_harmony_mismatch`: source global chroma `0.307`, active chroma `0.183`, onset correlation `0.552`, peak alignment `0.688`.

Interpretation: this simple source-chroma auxiliary loss is wired and measurable, but weight `1.0` does not solve active harmony on the paired smoke. The next harmony step should change harmonic target representation or model architecture.

Chroma-key architecture and best-checkpoint selection:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\paired_parser_exact_passed" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "paired_parser_exact_overfit20_chromakey_best" `
  --epochs 20 `
  --batch-size 1 `
  --hidden-channels 48 `
  --n-blocks 3 `
  --model-architecture chroma_key_conditioned `
  --key-embed-dim 24 `
  --source-onset-loss-weight 0.5 `
  --source-chroma-loss-weight 1.0 `
  --sample-every 5 `
  --sample-count 1 `
  --device cpu
```

Training now writes `checkpoints/best_sample_eval.pt` whenever an evaluated epoch improves the sample score, while still writing `latest.pt`. This matters because `paired_parser_exact_overfit20_chromakey_best` passed all gates at epoch 15, then regressed by epoch 20. The best checkpoint is epoch 15.

For paired caches, training sample evaluation is now target-aware when `index.csv` contains `target_midi`. The same target gates used by `validate-paired` are applied during epoch sampling, and target warnings affect both `sample_eval_pass_count` and `sample_eval_score`. The relevant CLI controls are `--no-target-eval`, `--min-target-global-chroma-cosine`, `--min-target-active-chroma-cosine`, and `--min-target-onset-correlation`.

Target-aware train smoke:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\paired_parser_exact_passed" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "paired_parser_exact_targeteval_train_smoke" `
  --epochs 1 `
  --batch-size 1 `
  --hidden-channels 32 `
  --n-blocks 2 `
  --model-architecture chroma_key_conditioned `
  --key-embed-dim 16 `
  --source-onset-loss-weight 0.5 `
  --source-chroma-loss-weight 1.0 `
  --sample-every 1 `
  --sample-count 1 `
  --device cpu
```

Evidence: `saves2\piano_arranger\runs\paired_parser_exact_targeteval_train_smoke` wrote target-aware sample eval, `history.json`, `latest.pt`, and `best_sample_eval.pt`. The one-epoch sample correctly failed with target warnings: target global chroma `0.122`, target active chroma `0.068`, target onset correlation `-0.088`, and `sample_eval_score=-69.900`. The sample MIDI parsed with 24 `note_on` events and the WAV loaded at 22.05 kHz.

Harmony-conditioned architecture smoke:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action train `
  --cache-dir "saves2\piano_arranger\cache\paired_parser_exact_passed" `
  --train-out-root "saves2\piano_arranger\runs" `
  --run-name "paired_parser_exact_harmonyconditioned10_smoke" `
  --epochs 10 `
  --batch-size 1 `
  --hidden-channels 48 `
  --n-blocks 3 `
  --model-architecture harmony_conditioned `
  --key-embed-dim 24 `
  --source-onset-loss-weight 0.5 `
  --source-chroma-loss-weight 0.5 `
  --harmonic-plan-loss-weight 2.0 `
  --sample-every 5 `
  --sample-count 1 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 32 `
  --min-unique-pitches 10 `
  --register-coverage-chunk-seconds 2 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 1 `
  --density-plan-peak-threshold 0.25 `
  --source-energy-velocity-weight 0.35 `
  --density-plan-velocity-weight 0.45 `
  --device cpu
```

Evidence: `paired_parser_exact_harmonyconditioned10_smoke` reduced loss from `4.697` to `3.974`, and `harmonic_plan_loss` from `0.119` to `0.114`. Epoch 10 still failed target-aware sample eval, but the warnings narrowed to source and target active-harmony mismatch: 42 notes, 10 unique pitches, source active chroma `0.176`, target active chroma `0.185`, source onset correlation `0.475`, target onset correlation `0.430`; MIDI/WAV loaded successfully. Batch validation at `saves2\piano_arranger\batch_eval\paired_parser_exact_harmonyconditioned10_targetgated` also failed 0/1 with the same active-harmony warnings. This is not a pass, but it is the first model-side harmonic-plan pressure test and moves target active chroma closer to the `0.20` gate.

Light harmonic-plan guided decoding can use the learned `pred["harmony"]` as a pitch-class scorer:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action validate-paired `
  --checkpoint "saves2\piano_arranger\runs\paired_parser_exact_harmonyconditioned10_smoke\checkpoints\best_sample_eval.pt" `
  --paired-manifest "data\piano_arranger_manifests\paired_parser_exact_discovered.audit.passed.csv" `
  --batch-out-dir "saves2\piano_arranger\batch_eval\paired_parser_exact_harmonyconditioned10_harmguide025_targetgated" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 32 `
  --min-unique-pitches 10 `
  --register-coverage-chunk-seconds 2 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 1 `
  --density-plan-peak-threshold 0.25 `
  --harmonic-plan-guidance-weight 0.25 `
  --source-energy-velocity-weight 0.35 `
  --density-plan-velocity-weight 0.45 `
  --device cpu
```

Evidence: `paired_parser_exact_harmonyconditioned10_harmguide025_targetgated` processed 1 reviewed row and failed only `source_active_harmony_mismatch`: source active chroma `0.198`, target active chroma `0.205`, source onset correlation `0.451`, target onset correlation `0.410`, 33 notes, 10 unique pitches, max simultaneity 8. MIDI/WAV loaded successfully, and arrangement metadata records harmonic-plan guidance as available with weight `0.25`. Stronger guidance `1.0` over-concentrated pitch classes; weaker guidance `0.15` missed the target-active gate.

Blend source-chroma and harmonic-plan guidance:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action validate-paired `
  --checkpoint "saves2\piano_arranger\runs\paired_parser_exact_harmonyconditioned10_smoke\checkpoints\best_sample_eval.pt" `
  --paired-manifest "data\piano_arranger_manifests\paired_parser_exact_discovered.audit.passed.csv" `
  --batch-out-dir "saves2\piano_arranger\batch_eval\paired_parser_exact_harmonyconditioned10_sourcechroma025_harmguide025_targetgated" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 32 `
  --min-unique-pitches 10 `
  --register-coverage-chunk-seconds 2 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 1 `
  --density-plan-peak-threshold 0.25 `
  --source-chroma-guidance-weight 0.25 `
  --harmonic-plan-guidance-weight 0.25 `
  --source-energy-velocity-weight 0.35 `
  --density-plan-velocity-weight 0.45 `
  --device cpu
```

Evidence: `paired_parser_exact_harmonyconditioned10_sourcechroma025_harmguide025_targetgated` is the first target-gated reviewed-pair pass: processed 1 row, passed 1/1, no warnings. Metrics were 44 notes, 11 unique pitches, max simultaneity 12, source global chroma `0.548`, source active chroma `0.319`, source onset correlation `0.560`, target global chroma `0.459`, target active chroma `0.347`, and target onset correlation `0.430`. MIDI parsed with 44 `note_on` events, WAV loaded at 22.05 kHz, and metadata confirms both source-chroma and harmonic-plan guidance were available with weight `0.25`. This is parser-smoke evidence only; the next real test is a larger reviewed paired manifest.

Evidence: the chroma-key run reduced loss from `4.616` to `3.363` and source-chroma loss from `0.0823` to `0.0796`. Epoch 15 sample passed with active chroma `0.257`, onset correlation `0.440`, and peak alignment `0.909`; epoch 20 failed active harmony again.

Standalone inference from `checkpoints\best_sample_eval.pt` rendered `saves2\piano_arranger\outputs\paired_parser_exact_overfit20_chromakey_best\1390_overfit20_chromakey_best_densityplan.*`. MIDI parsed with 41 `note_on` events and WAV loaded at 22.05 kHz. Eval report `saves2\piano_arranger\reports\paired_parser_exact_overfit20_chromakey_best_densityplan_source_eval.json` passed with no warnings: 41 notes, 12 unique pitches, source global chroma `0.449`, active chroma `0.350`, onset correlation `0.458`, peak alignment `0.917`.

This is the first paired-smoke evidence that the DGGR-native source-audio-to-piano-roll model can pass both rhythm and active-harmony gates. It is still one reviewed parser pair, not real-corpus or full-song quality.

Validate a paired checkpoint across a reviewed manifest:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action validate-paired `
  --checkpoint "saves2\piano_arranger\runs\paired_parser_exact_overfit20_chromakey_best\checkpoints\best_sample_eval.pt" `
  --paired-manifest "data\piano_arranger_manifests\paired_parser_exact_discovered.audit.passed.csv" `
  --batch-out-dir "saves2\piano_arranger\batch_eval\paired_parser_exact_chromakey_best_targetgated" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 32 `
  --min-unique-pitches 10 `
  --register-coverage-chunk-seconds 2 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 1 `
  --density-plan-peak-threshold 0.25 `
  --source-energy-velocity-weight 0.35 `
  --density-plan-velocity-weight 0.45 `
  --device cpu
```

Batch validation writes generated artifacts under `outputs/`, per-row reports under `reports/`, a `rows.csv`, and `summary.json`. It evaluates both source alignment and target-MIDI alignment by default. The default paired target gates are `min_target_global_chroma_cosine=0.20`, `min_target_active_chroma_cosine=0.20`, and `min_target_onset_correlation=0.02`; use `--no-target-eval` only for source-only debugging.

Paired target diagnostics also include non-gating onset match metrics: `target_onset_frame_precision/recall/f1`, `target_pitch_class_onset_precision/recall/f1`, and `target_note_count_ratio`. These use a +/-1 frame tolerance. Keep them visible in reports because they expose exact supervised target-fidelity gaps that the chroma gates can miss.

Optional strict target-event gates are available: `--min-target-onset-frame-f1`, `--min-target-pitch-class-onset-f1`, `--min-target-note-count-ratio`, and `--max-target-note-count-ratio`. Their defaults are `0.0`, which disables them for parser-smoke compatibility.

Target-gated parser exact smoke evidence: `saves2\piano_arranger\batch_eval\paired_parser_exact_chromakey_best_targetgated` processed 1 reviewed row and passed 0/1 because of `target_active_harmony_mismatch`. Mean metrics were 33 notes, 12 unique pitches, max simultaneity 7, source global chroma `0.481`, source active chroma `0.319`, source onset correlation `0.320`, source peak alignment `0.818`, target global chroma `0.227`, target active chroma `0.129`, and target onset correlation `0.496`; MIDI parsed with 33 `note_on` events and WAV loaded at 22.05 kHz. This is a better signal than the source-only pass: the checkpoint creates valid, source-aligned piano structure, but does not yet reproduce the paired target harmony strongly enough.

Chunked paired validation uses the same action plus `--batch-chunked`; it runs chunked inference, writes a per-row section report, and treats section warnings as validation warnings:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action validate-paired `
  --batch-chunked `
  --checkpoint "saves2\piano_arranger\runs\paired_parser_exact_harmonyconditioned10_smoke\checkpoints\best_sample_eval.pt" `
  --paired-manifest "data\piano_arranger_manifests\paired_parser_exact_discovered.audit.passed.csv" `
  --batch-out-dir "saves2\piano_arranger\batch_eval\paired_parser_exact_harmonyconditioned10_chunked2s_sourcechroma025_harmguide025_targetgated" `
  --seconds 4 `
  --max-frames 128 `
  --frame-hz 25 `
  --chunk-seconds 2 `
  --chunk-hop-seconds 2 `
  --section-profile flat `
  --section-seconds 2 `
  --max-onsets-per-frame 4 `
  --max-note-duration 1.5 `
  --min-selected-notes 32 `
  --min-unique-pitches 10 `
  --register-coverage-chunk-seconds 2 `
  --density-plan-guidance-weight 1.0 `
  --density-plan-snap-frames 1 `
  --density-plan-peak-threshold 0.25 `
  --source-chroma-guidance-weight 0.25 `
  --harmonic-plan-guidance-weight 0.25 `
  --source-energy-velocity-weight 0.35 `
  --density-plan-velocity-weight 0.45 `
  --device cpu
```

Chunked parser-pair smoke evidence: `paired_parser_exact_harmonyconditioned10_chunked2s_sourcechroma025_harmguide025_targetgated` processed 1 reviewed row and passed 1/1 with no warnings. It used 2 chunks, produced 49 notes and 14 unique pitches, source active chroma `0.349`, target active chroma `0.313`, source onset correlation `0.470`, target onset correlation `0.386`, target onset-frame F1 `0.533`, target pitch-class-onset F1 `0.289`, and target note-count ratio `1.021`. The section report had 2 sections, no warnings, min section notes `21`, max section notes `28`, and min high-register fraction `0.464`. MIDI/WAV loaded successfully. This is parser-smoke chunking evidence, not full-song proof. The low pitch-class-onset F1 means the next architecture step should improve chord/voicing/event representation, not just preserve this decode recipe.

Musical-plan supervision smoke: `saves2\piano_arranger\cache\paired_parser_exact_musicalplan_passed` wrote `target_chord (1,13,128)`, `target_bass (1,13,128)`, and `target_voicing (1,4,128)`. `saves2\piano_arranger\runs\paired_parser_exact_musicalplan_train_smoke` ran one CPU batch with `--model-architecture harmony_conditioned`, `--harmonic-plan-loss-weight 0.5`, and `--musical-plan-loss-weight 0.5`; history logged `musical_plan_loss=3.578` and wrote `checkpoints\latest.pt`. This proves the new heads and loss are wired, not that they improve listening quality yet.

Musical-plan decode smoke: `saves2\piano_arranger\batch_eval\paired_parser_exact_harmonyconditioned10_chunked2s_plancompat_targetgated` passed 1/1 with the older harmony-conditioned checkpoint and nonzero musical-plan guidance weights; metadata correctly marked chord/bass/voicing unavailable because that checkpoint lacks the new heads. `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplan_plan_guidance025_targetgated` ran the one-batch musical-plan checkpoint with plan guidance available and failed 0/1 only on source rhythm: target active chroma `0.305`, target onset correlation `0.059`, target onset-frame F1 `0.278`, pitch-class-onset F1 `0.049`. This is end-to-end wiring evidence and a useful negative result; real training must improve event-level F1 before quality claims.

Musical-plan-conditioned architecture smoke: `saves2\piano_arranger\runs\paired_parser_exact_musicalplanconditioned5_smoke` trained 5 CPU epochs with `--model-architecture musical_plan_conditioned`, `--harmonic-plan-loss-weight 0.5`, and `--musical-plan-loss-weight 0.5`. Loss decreased from `6.805` to `5.368`; `musical_plan_loss` decreased from `3.783` to `2.319`; best sample-eval epoch was 5. Target-gated validation at `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_plan_guidance025_targetgated` failed 0/1 with `harmony_collapse` and `source_rhythm_mismatch`: 32 notes, 12 unique pitches, source active chroma `0.690`, target active chroma `0.411`, target onset correlation `0.048`, target onset-frame F1 `0.400`, pitch-class-onset F1 `0.125`. This is a useful architecture result, not a pass: direct plan conditioning improves coarse harmony/event metrics but currently concentrates pitch class and loses rhythm.

Event-plan and rebalanced-decode smoke: `saves2\piano_arranger\cache\paired_parser_exact_eventplan_passed` wrote `target_event (1,4,128)`. `saves2\piano_arranger\runs\paired_parser_exact_musicalplanconditioned5_eventanti_smoke` trained 5 CPU epochs with event-plan weight `0.75` and anti-collapse weight `0.5`; loss dropped `7.060 -> 5.804`, `event_plan_loss` dropped `0.452 -> 0.326`, and `musical_plan_loss` dropped `3.441 -> 2.219`. Target-gated validation at `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_eventanti_rebalanced_targetgated` failed 0/1 only with `source_rhythm_mismatch`: 32 notes, 10 unique pitches, single pitch-class fraction `0.3125`, target active chroma `0.322`, target onset correlation `0.039`, target onset-frame F1 `0.235`, pitch-class-onset F1 `0.100`. This removed `harmony_collapse`, but event timing is still weak; the next step should use event/source peaks to drive start selection more directly.

Event/source-snapped decode smoke: `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_eventsource_snap_targetgated` used the event/anti best checkpoint with source-onset snapping plus `--event-plan-guidance-weight 1.0 --event-plan-snap-frames 2 --event-plan-peak-threshold 0.25`. It passed 1/1 with no warnings: 32 notes, 10 unique pitches, max simultaneity `10`, single pitch-class fraction `0.3125`, source onset correlation `0.288`, target active chroma `0.342`, target onset correlation `0.033`, target onset-frame F1 `0.323`, pitch-class-onset F1 `0.050`, and target note-count ratio `0.667`. Metadata showed source onset, event-plan, and density-plan snapping all active. This is the current best parser-pair result for the newer plan-conditioned path, but the low pitch-class-onset F1 and note-count ratio mean event-level target fidelity remains weak.

Fuller event/source-snapped decode smoke: `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_eventsource_snap_min48_regfix_targetgated` used the same checkpoint with `--min-selected-notes 48 --min-unique-pitches 12` plus register rebalance repair. It passed 1/1 with no warnings: 48 notes, 14 unique pitches, max simultaneity `12`, single pitch-class fraction `0.3125`, register fractions low/mid/high `0.521/0.417/0.0625`, source onset correlation `0.459`, target active chroma `0.362`, target onset correlation `0.084`, target onset-frame F1 `0.444`, pitch-class-onset F1 `0.167`, and target note-count ratio `1.000`. MIDI parsed with 48 `note_on` events. This is the best parser-pair evidence so far for fullness plus rhythm, but it is still one reviewed parser pair, not a real-corpus/full-song result.

Fuller chunked event/source-snapped smoke: `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_chunked2s_eventsource_snap_min48_regfix_targetgated` used `--batch-chunked --chunk-seconds 2 --chunk-hop-seconds 2 --section-seconds 2` and passed 1/1 with no warnings. It produced 48 notes across 2 chunks, 13 unique pitches, max simultaneity `12`, single pitch-class fraction `0.292`, source onset correlation `0.681`, source peak alignment `0.619`, target active chroma `0.581`, target onset correlation `0.044`, target onset-frame F1 `0.556`, pitch-class-onset F1 `0.250`, and target note-count ratio `1.000`. MIDI parsed with 48 `note_on` events. The section report had no warnings; both chunks kept 24 notes. This is the current strongest parser-pair evidence for the newer plan-conditioned path, but it still needs a real reviewed paired manifest and longer full-song validation.

Strict target-event smoke: `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_chunked2s_eventsource_snap_min48_regfix_strictf1` reran the same chunked decode with `--min-target-onset-frame-f1 0.50 --min-target-pitch-class-onset-f1 0.35 --min-target-note-count-ratio 0.85 --max-target-note-count-ratio 1.20`. It failed 0/1 only with `target_pitch_class_onset_f1_mismatch`: onset-frame F1 `0.556` and note-count ratio `1.000` passed, but pitch-class-onset F1 was only `0.250`. This is the current honest gap after the parser-smoke pass.

Pitch-class-onset plan smoke: `saves2\piano_arranger\cache\paired_parser_exact_pconset_passed` wrote `target_pc_onset (1,12,128)`. `saves2\piano_arranger\runs\paired_parser_exact_musicalplanconditioned5_pconset_smoke` trained 5 CPU epochs with `--pc-onset-plan-loss-weight 1.0`; `pc_onset_plan_loss` decreased `1.416 -> 1.121`. Strict validation with `--pc-onset-plan-guidance-weight 0.10` at `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_pconset_pcguide010_strictf1` improved pitch-class-onset F1 to `0.3125` with no collapse warnings, but still missed the strict `0.35` gate. Stronger pc-onset guidance reached `0.3333` and reintroduced `harmony_collapse`, so this is architecture progress, not a solved pass.

Collapse-aware pc-event distribution smoke: `piano_roll_loss` now logs `pc_onset_distribution_loss`, adds pc-event frame/global distribution pressure to the pc-onset plan objective, and includes pc-event entropy/dominance in `anti_collapse_loss`. Register rebalance now respects `max_pitch_class_fraction` so it cannot undo pitch-class pruning. `saves2\piano_arranger\runs\paired_parser_exact_musicalplanconditioned5_pconset_eventdist_smoke` trained 5 CPU epochs and logged `pc_onset_distribution_loss 1.445 -> 1.209`. Strict validation after the register-cap fix at `saves2\piano_arranger\batch_eval\paired_parser_exact_musicalplanconditioned5_pconset_eventdist_pcguide010_regcap_strictf1` failed only the strict pitch-class-onset gate: 48 notes, 16 unique pitches, single pitch-class fraction `0.292`, source onset correlation `0.803`, target onset-frame F1 `0.545`, pitch-class-onset F1 `0.3333`, and note-count ratio `1.000`. The current strict target is still `0.35`, so the gap is narrower but not closed. `--pc-onset-f1-loss-weight` is wired as a model-side test because it optimizes a soft version of the strict +/-1-frame pitch-class-onset F1 metric rather than only BCE or distribution shape. Wiring smoke `saves2\piano_arranger\runs\paired_parser_exact_pconset_f1_wiring_smoke` ran 1 CPU batch with `--pc-onset-f1-loss-weight 0.5` and logged `pc_onset_f1_loss 0.8950`; sample eval failed, so this is objective plumbing evidence only. The first short comparison is negative: `saves2\piano_arranger\runs\paired_parser_exact_pconset_f1_eventdist5_smoke` trained 5 CPU epochs with `--pc-onset-f1-loss-weight 0.5`, but strict chunked validation at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_f1_eventdist5_pcguide010_regcap_strictf1` failed only `target_pitch_class_onset_f1_mismatch` with pitch-class-onset F1 `0.2708`, below the previous `0.3333` best. Direct note-head pitch-class alignment is also wired via `--pc-onset-alignment-loss-weight`; `saves2\piano_arranger\runs\paired_parser_exact_pconset_align005_eventdist5_smoke` trained 5 CPU epochs with weight `0.05` and reduced `pc_onset_alignment_loss 3.359 -> 2.464`, but strict validation at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_align005_eventdist5_pcguide010_regcap_strictf1` reached only pitch-class-onset F1 `0.2292`. Do not repeat either scalar-only setup as the next candidate.

Pc-onset reserve decode smoke: using the prior event-distribution best checkpoint with strict chunked validation, reserve threshold `0.50` and max-per-frame `2` at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcreserve050_pcguide010_regcap_strictf1` reserved 31/34 candidates across the two chunks, improved target onset-frame F1 to `0.625`, and failed only pitch-class-onset F1 at `0.3125`. Reserve threshold `0.65` and max-per-frame `1` at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcreserve065_pcguide010_regcap_strictf1` reserved 9/12 candidates, matched pitch-class-onset F1 `0.3333`, but added `single_pitch_collapse`. This decode mechanism is wired and useful for diagnosing selected-event behavior, but it is not a new best.

Protected pc-onset selected-event reservation is also wired through `--pc-onset-plan-select-reserve-fraction`. With reserve threshold `0.65`, max-per-frame `1`, and select fraction `0.50`, `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcreserve065_select050_strictf1` overfilled to 64 notes and failed `target_note_count_overfill` with pitch-class-onset F1 `0.2679`. Tightening decode to `--max-notes-per-second 12` at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcreserve065_select050_nps12_strictf1` restored 48 notes and note-count ratio `1.000`, but still failed pitch-class-onset F1 at `0.3125`. This is useful negative evidence: protected reserve can enforce plan events under count control, but the next architecture step needs count-aware event-to-note assignment/ranking rather than another reserve threshold sweep.

Count-aware pc-onset event-to-note assignment is now wired through `--pc-onset-plan-assign-threshold`, `--pc-onset-plan-assign-fraction`, `--pc-onset-plan-assign-window-frames`, and `--pc-onset-plan-assign-min-note-score`. It assigns learned pc-onset local maxima to same-pitch-class notes inside a small timing window before normal decode fill. Half-budget assignment at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcassign065_frac050_nps12_strictf1` selected 12 assigned notes per chunk and preserved note count, but failed pitch-class-onset F1 at `0.2917`. Full-budget assignment at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcassign065_frac100_nps12_strictf1` passed the strict parser-pair gates with 48 notes, 16 unique pitches, source active chroma `0.622`, source onset correlation `0.647`, target active chroma `0.489`, target onset-frame F1 `0.629`, pitch-class-onset F1 `0.3958`, target note-count ratio `1.000`, no global warnings, and no section warnings. This beats the previous `0.3333` parser-pair best, but it is still one reviewed parser pair and must not be treated as real-corpus/full-song proof.

Source/event-aware assignment ranking is wired through `--pc-onset-plan-assign-source-weight`, `--pc-onset-plan-assign-event-weight`, and `--pc-onset-plan-assign-distance-penalty`. Defaults preserve the older distance-first assignment behavior. Source-aware strict parser-pair validation at `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_pcassign065_frac100_nps12_srcaware_w2_strictf1` used window `2`, source weight `2.0`, event weight `1.0`, and distance penalty `0.25`; it passed 1/1 with pitch-class-onset F1 `0.3958`, target onset-frame F1 `0.710`, source active chroma `0.644`, source onset correlation `0.680`, note-count ratio `1.000`, and no warnings.

Chunked combined-output bass repair now runs when `--section-bass-repair` is enabled. This repairs sections after overlap/hop trimming, because per-chunk bass repairs can be cropped out of the kept region. Metadata records `post_chunk_section_bass_repairs`.

Longer assignment render evidence: `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassrepair_postbass_pitchcap18_pcassign065_frac100_sim10_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` used the strong roletexture-melody-texturerole checkpoint with full pc-onset assignment, max simultaneity `10`, pitch cap `0.18`, and post-combination bass repair. It passed global and section eval with no warnings: 291 notes, 28 unique pitches, source active chroma `0.426`, source onset correlation `0.487`, source peak alignment `0.784`, chord-frame `0.895`, bass `0.459`, melody `0.341`, mean polyphony `5.905`, fullness `1.000`, RMS `0.205`, min section notes `28`, min section bass `0.120`, min section chord `0.700`, min section fullness `0.767`, and mean section fullness `0.925`. Compared with the older bass-repaired baseline, this is fuller and more top-line/body-rich, but it sacrifices source onset correlation (`0.601 -> 0.487`) and a little RMS (`0.216 -> 0.205`). Half assignment was not better for this long render because it kept `section_bass_absent:0` and dropped to 231 notes, chord-frame `0.773`, melody `0.256`, mean polyphony `4.883`, and RMS `0.189`.

Source-aware longer assignment evidence: `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassrepair_postbass_pitchcap18_pcassign065_frac100_srcaware_w2_sim10_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` passed global and section eval with no warnings. It keeps 291 notes, improves to 29 unique pitches, source active chroma `0.436`, source onset correlation `0.561`, chord-frame `0.920`, bass `0.496`, mean polyphony `6.195`, fullness `0.993`, RMS `0.200`, min section notes `31`, min section chord `0.830`, min section fullness `0.772`, and mean section fullness `0.927`. It is the better balance when source timing matters, but it lowers melody coverage versus non-source-aware assignment (`0.341 -> 0.289`) and is still not real paired/full-song proof.

Source-only audit harness: `--action audit-sources` now accepts a source manifest with `source_audio`, `path`, or `audio_path`, renders every row, runs source-aware eval plus section reporting, measures preview WAV RMS/peak, and writes `rows.csv` plus `summary.json`. First smoke manifest: `data\piano_arranger_manifests\source_audit_two_sources.csv`. First audit output: `saves2\piano_arranger\source_audit\two_sources_srcaware_24s`.

Two-source source-aware assignment audit: the current 24-second source/event-aware recipe processed the Hui-style and Moonlight-style sources, passed 1/2, and failed only `section_pitch_underuse:0` on the Moonlight-style source. Aggregate metrics were strong: mean notes `289.5`, unique pitches `28.0`, source active chroma `0.551`, source onset correlation `0.569`, source peak alignment `0.670`, chord-frame `0.858`, bass `0.678`, melody `0.342`, mean active polyphony `5.908`, fullness `0.996`, WAV RMS `0.200`, peak `0.950`. This is broader than the single Hui render, but it exposes the next pressure point: prevent weak/under-diverse opening sections while retaining fullness and source timing.

Section diversity repair audit: `saves2\piano_arranger\source_audit\two_sources_srcaware_24s_sectiondivrepair` adds the opt-in post-chunk guardrail `--section-diversity-repair --section-diversity-repair-min-unique-pitches 8 --section-diversity-repair-min-chord-frame 0.15 --section-diversity-repair-max-notes 4`. It passed 2/2 with no warnings. The Hui row needed no diversity repair; the Moonlight row added 3 notes and fixed the opening section (`min_section_notes 12 -> 15`, `min_section_chord_frame 0.090 -> 0.340`, `min_section_fullness 0.580 -> 0.779`) while keeping source onset correlation essentially stable (`0.578 -> 0.574`). The rerun now records `min_section_unique_pitches`; both rows have min `9`, above the new checkpoint selector target of `8`. Treat this as decode scaffolding and a model target: the next real architecture step should learn section-local body/diversity rather than depend on post-hoc insertion.

Learned section-diversity decode guidance is now wired as that next experiment hook. Train samples, inference, chunked inference, paired validation, and source audit expose `--section-diversity-guidance-weight`, `--section-diversity-reserve-fraction`, `--section-diversity-reserve-min-note-score`, `--section-diversity-unique-weight`, `--section-diversity-pc-weight`, `--section-diversity-range-weight`, `--section-diversity-onset-weight`, and `--section-diversity-section-seconds`. When enabled, decode reads the loaded `section_diversity [4,T]` head and can reserve a bounded fraction of protected notes that add local unique pitches, pitch classes, and low/high range before normal note fill. Compile passed, CLI help exposes the flags, and a synthetic smoke reserved 4 learned-guidance notes, but this is plumbing evidence only; it still needs real checkpoint/source-audit evidence.

Real source-audit guidance diagnostics are negative but useful. The current best musical checkpoint does not contain `section_diversity_head`, so learned guidance cannot affect it. The weak headed checkpoint `saves2\piano_arranger\runs\section_unique_selector_3epoch_smoke\checkpoints\best_sample_eval.pt` was audited only as a diagnostic. No-guidance output `saves2\piano_arranger\source_audit\two_sources_sectionunique3_noguidance_diag` failed 0/2 with mean unique pitches `8.5`, source onset correlation `0.086`, min section unique `7`, and section pitch-underuse warnings. Light guidance `two_sources_sectionunique3_guidance025_reserve010_diag` used the learned head in all 24 chunks and reserved 24 notes; it still failed 0/2, improved some body/RMS metrics, but kept min section unique at `7` and introduced a single-pitch collapse warning. Strong guidance `two_sources_sectionunique3_guidance100_reserve025_diag` reserved 48 notes and failed worse, with min section unique `6`, min section chord `0.200`, and more collapse/section warnings. Do not keep auditing weak selector checkpoints as candidates.

Warm-start training is now the intended bridge. `--warm-start-checkpoint` loads matching checkpoint tensors and leaves new heads initialized. Smoke `saves2\piano_arranger\runs\warmstart_sectiondiv_guidance_wiring_smoke` warm-started from the strong roletexture checkpoint, loaded 105 keys, left 20 newer section/arranger/diversity keys initialized, ran one CUDA batch, logged `section_diversity_plan_loss 0.264`, and wrote a checkpoint containing `section_diversity_key_proj` plus `section_diversity_head`. The next serious run should warm-start from the strong roletexture checkpoint, train section-diversity/section-health heads at low weight, then audit with light learned guidance against the repair-only two-source baseline.

First warm-start section-diversity run: `saves2\piano_arranger\runs\warmstart_roletexture_sectiondiv005_guidance_ready_3epoch` used the strong roletexture checkpoint, the same role/texture/melody/texturerole recipe, and low section-diversity weights (`0.05` plan, `0.025` balance). Over 3 CUDA epochs, loss fell `5.423 -> 5.112`, `section_diversity_plan_loss 0.274 -> 0.223`, and sample score improved `-117.715 -> -57.443`; best sample checkpoint was epoch 3. Samples still failed, so this is not a candidate by itself, but it is the first headed warm-start branch with real training signal.

Two-source audits for that warm-start branch are mixed. Baseline source-aware decode with no learned section-diversity guidance, `saves2\piano_arranger\source_audit\two_sources_warmstart_sectiondiv005_noguidance`, passed 1/2 and failed only Moonlight overstacking. Light learned guidance `two_sources_warmstart_sectiondiv005_guidance025_reserve010` used the learned head in all chunks and reserved about 23 notes per row, but failed 0/2; it reintroduced Hui section pitch-underuse and kept Moonlight overstacking. Tighter no-guidance decode `two_sources_warmstart_sectiondiv005_noguidance_sim8` passed 2/2 with `--max-simultaneous-notes 8`, 195.5 mean notes, 33 mean unique pitches, source onset corr `0.384`, chord `0.358`, bass `0.236`, melody `0.747`, mean polyphony `3.654`, fullness `0.845`, RMS `0.138`, min section unique `8`, and no warnings. Treat this as proof the warm-start headed branch can pass the two-source gate, not as a final "oomph" model; it is far thinner and quieter than `two_sources_srcaware_24s_sectiondivrepair`.

Body-biased warm-start result: `saves2\piano_arranger\runs\warmstart_roletexture_bodybiased_sectiondiv001_quality_3epoch` lowered section-diversity pressure (`0.01` plan, no balance), strengthened texture/body pressure, and used stricter quality scoring. Best checkpoint was epoch 1; later epochs became thinner. Audit `two_sources_warmstart_bodybiased_sectiondiv001_noguidance` recovered the oomph side: mean notes `572.5`, unique pitches `25.5`, source onset corr `0.773`, chord `0.972`, bass `0.539`, melody `0.878`, polyphony `8.876`, fullness `1.000`, RMS `0.216`, min section unique `13`, min section chord `0.760`, min section fullness `0.891`. It passed Hui but failed Moonlight with `overstacked` and `mid_harmony_underuse`. Tighter decode `two_sources_warmstart_bodybiased_sectiondiv001_noguidance_sim8` removed overstacking but still failed Moonlight mid-harmony underuse. Disabling voicing guidance made mid-register underuse worse on both rows, so the next model needs selector/training pressure for mid-body balance, not just a voicing decode tweak.

Mid-body/high-register selector pressure is now wired. Training sample eval aggregates `mid_note_fraction` and `high_note_fraction`; the quality score logs `mid_note_penalty` and `high_note_penalty`, controlled by `--sample-score-min-mid-note-fraction` and `--sample-score-max-high-note-fraction`. Compile passed, CLI help exposes both flags, and a direct score smoke with `mean_mid_note_fraction=0.08` plus `mean_high_note_fraction=0.84` produced `mid_note_penalty=0.600` and `high_note_penalty=0.360`.

Mid-selector warm-start evidence is negative at the first tested setting. `saves2\piano_arranger\runs\warmstart_bodybiased_midselector_sectiondiv001_3epoch_s1` used the body-biased recipe plus stricter sample scoring (`--sample-score-quality-penalty-weight 65 --sample-score-min-mid-note-fraction 0.20 --sample-score-max-high-note-fraction 0.75`). The penalties fired during training, but best sample-eval still selected epoch 1; later epochs became more high-register dominated. Source audit `saves2\piano_arranger\source_audit\two_sources_warmstart_midselector_sectiondiv001_s1_sim8` passed 1/2 and failed Moonlight with the same `mid_harmony_underuse` warning as the body-biased `sim8` baseline. Its deployment metrics are effectively unchanged: mean notes `566`, unique `25`, bass/mid/high note fractions `0.084/0.191/0.726`, source onset corr `0.752`, chord `0.945`, bass `0.453`, melody `0.885`, polyphony `7.221`, fullness `0.988`, RMS `0.205`, min section unique `12`. The Moonlight row is the actual failure mode: mid `0.108`, high `0.788`. Do not keep rerunning selector-only warm starts; the next candidate needs model-side mid/body supervision or decode instrumentation that can prove whether the checkpoint is exposing usable mid-register candidates.

Body/melody-state mid-body warm-start is partial but still negative. `saves2\piano_arranger\runs\warmstart_bodybiased_bodymelody_midbody_sectiondiv001_3epoch_s1` added `--body-melody-state-plan-loss-weight 0.10 --body-melody-state-balance-loss-weight 0.30` to the same body-biased + selector recipe. Losses decreased, but sample eval still selected epoch 1 and later epochs became more high-register dominated. Source audit `saves2\piano_arranger\source_audit\two_sources_warmstart_bodymelody_midbody_sectiondiv001_s1_sim8` passed 1/2 and still failed Moonlight `mid_harmony_underuse`. It nudged the aggregate register balance in the right direction compared with selector-only (`mid 0.191 -> 0.201`, high `0.726 -> 0.707`) while preserving fullness (`0.993`) and source onset (`0.751`), but the failing Moonlight row only reached mid `0.127`, high `0.766`. This means existing `body_melody_state_balance_loss` has useful signal but is too weak or too indirect as currently weighted. Next experiment should either strengthen mid-body pressure with a targeted middle-register floor/over-high penalty in the loss, or inspect decode candidates to see whether mid-register notes are present but losing selection.

Pc-onset plan diagnostic evidence: `saves2\piano_arranger\batch_eval\paired_parser_exact_pconset_eventdist_plan_diag_strictf1` used the current event-distribution best checkpoint with the strict chunked recipe and no reserve decode. Decoded notes had pitch-class-onset F1 `0.2708`, but the raw pc-onset plan reached best F1 `0.4127` at threshold `0.65` with precision `0.3333`, recall `0.5417`, 78 predicted local-max events, and 48 target events. That means the checkpoint contains enough pc-event signal in the plan head; the failing part is selected-note ranking/assignment under decode constraints. The next serious model step should convert plan events into selected notes more directly instead of adding more scalar pressure to the same plan.

Validation plumbing note: `validate-paired` now forwards `section_bass_repair` and `section_bass_repair_min_coverage` through `PairedCheckpointBatchEvalConfig`, matching the CLI arguments used by infer/chunked infer.

Broader pseudo-pretraining smoke: `saves2\piano_arranger\cache\bootstrap_16track_8s_planheads` was built from 16 rows of `piano_candidates.csv`, 8-second windows, and wrote all current target heads including `target_event (16,4,256)` and `target_pc_onset (16,12,256)`. The default-LR run `bootstrap_16track_8s_musicalplan_eventdist_smoke` collapsed after epoch 1 despite falling loss. The low-LR run `saves2\piano_arranger\runs\bootstrap_16track_8s_musicalplan_eventdist_lowlr_smoke` with `--lr 5e-4` trained stably for 5 epochs: loss `8.002 -> 6.633`, `musical_plan_loss 3.413 -> 2.345`, `event_plan_loss 0.474 -> 0.346`, `pc_onset_plan_loss 1.627 -> 1.361`, and `pc_onset_distribution_loss 1.157 -> 1.061`. Sample eval passed 2/2 with no warnings from epochs 2-5; best epoch was 5.

Role/fullness plan plumbing: new caches now write `target_role (N,5,T)` for bass activity, chord-frame activity, melody-register activity, normalized polyphony, and active-frame velocity weight. Older caches derive the same tensor on load, so existing `bootstrap_16track_8s_planheads` remains usable. `PianoRollGenerator` emits `role`, `musical_plan_conditioned` injects it into per-key note heads, and `piano_roll_loss` logs `role_plan_loss` when `--role-plan-loss-weight` is nonzero. Smoke evidence: `saves2\piano_arranger\cache\bootstrap_1track_8s_roleplan_smoke` built 1 row with `target_role_shape [5,256]`; `saves2\piano_arranger\runs\bootstrap_16track_8s_roleplan_smoke` ran 1 CUDA batch on the older 16-track cache and logged `role_plan_loss 0.4306`. Old-checkpoint inference zero-initializes missing `role_head` / `role_key_proj` modules so earlier `musical_plan_conditioned` checkpoints are not changed by random new role weights. Fresh `musical_plan_conditioned` models also zero-initialize `role_key_proj`, making role injection neutral at startup.

Role-plan low-LR comparison: `saves2\piano_arranger\runs\bootstrap_16track_8s_roleplan_lowlr_smoke` trained 5 CUDA epochs and reduced `role_plan_loss 0.420 -> 0.343`, but its 24s artifact was bass-heavy and failed global eval with `single_pitch_collapse` plus `mid_harmony_underuse`. After the neutral role-injection init fix, `saves2\piano_arranger\runs\bootstrap_16track_8s_roleplan_neutral_lowlr_smoke` trained 5 CUDA epochs, reduced `role_plan_loss 0.419 -> 0.340`, and selected epoch 3 as best sample-eval checkpoint. The velocity-boost artifact `saves2\piano_arranger\outputs\bootstrap_16track_8s_roleplan_neutral_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roleplan_neutral_velboost_model_piano.{json,mid,wav}` passed global and section eval with 111 notes, 17 unique pitches, source active chroma `0.504`, source onset correlation `0.567`, source peak alignment `0.887`, bass coverage `0.938`, chord-frame fraction `0.843`, velocity std `6.034`, and RMS `0.198`. This is stronger and more source-chroma/fullness-weighted than the non-role artifact, but it is less dense and more bass-heavy.

Texture/body balance comparison: `saves2\piano_arranger\runs\bootstrap_16track_8s_roleplan_texture_lowlr_smoke` trained 5 CUDA epochs with `--role-plan-loss-weight 0.5 --texture-balance-loss-weight 0.5`; `texture_balance_loss 1.746 -> 1.656`, `role_plan_loss 0.420 -> 0.342`, best sample-eval checkpoint epoch 4. The artifact `saves2\piano_arranger\outputs\bootstrap_16track_8s_roleplan_texture_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roletexture_velboost_model_piano.{json,mid,wav}` passed global and section eval with 168 notes, 26 unique pitches, source active chroma `0.489`, source onset correlation `0.557`, source peak alignment `0.845`, mid note fraction `0.476`, bass coverage `0.733`, chord-frame fraction `0.925`, velocity std `6.732`, and RMS `0.224`. This recovers note density and mid-register body while keeping strong fullness; the remaining tradeoff is lower top-line/melody coverage.

Melody/top-line planning comparison: `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_lowlr_smoke` trained 5 CUDA epochs with role, texture, and melody weights all at `0.5`; `melody_plan_loss 0.371 -> 0.261`, while `melody_balance_loss` stayed roughly flat near `0.88`, best sample-eval checkpoint epoch 2. The artifact `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roletexture_melody_velboost_model_piano.{json,mid,wav}` passed global and section eval with 180 notes, 25 unique pitches, melody coverage `0.531`, bass coverage `0.759`, chord-frame fraction `0.742`, source active chroma `0.439`, source onset correlation `0.504`, source peak alignment `0.744`, velocity std `8.980`, fullness score `1.000`, and RMS `0.185`. Compared with role+texture, it greatly improves melody coverage (`0.130 -> 0.531`) but weakens chord-frame density, mean polyphony, source fidelity, and loudness/body.

Joint texture-role planning is now wired for the next model step. New caches write `target_texture_role (N,4,T)` for bass floor, chord body, inner motion, and top-line presence; older caches derive it on load. `PianoRollGenerator` emits `texture_role`, `musical_plan_conditioned` injects it with a zero-initialized projection, and `piano_roll_loss` logs `texture_role_plan_loss` plus `texture_role_balance_loss`. Backward-compatibility smoke on old `bootstrap_16track_8s_planheads` derived `texture_role (4,256)` despite no `target_texture_role.npy`; after the normalized role-mix balance refinement, direct loss smoke produced `texture_role_plan_loss 0.481` and `texture_role_balance_loss 0.284`. Old-checkpoint inference compatibility was checked on the role+texture checkpoint; missing `texture_role_head` / `texture_role_key_proj` modules were zero-initialized.

Section-role planning is now wired for the learned section-continuity step. New caches write `target_section_role (N,4,T)` for local section bass coverage, chord body, melody presence, and fullness; older caches derive it on load using cache `frame_hz`. `PianoRollGenerator` emits `section_role`, `musical_plan_conditioned` injects it with a zero-initialized projection, and `piano_roll_loss` logs `section_role_plan_loss` plus `section_role_balance_loss`. Backward-compatible dataset/model/loss smoke on old `bootstrap_16track_8s_planheads` derived `section_role (1,4,256)`, predicted `section_role (1,4,256)`, and produced `section_role_plan_loss 0.419` plus `section_role_balance_loss 0.131`. Public CLI smoke `saves2\piano_arranger\runs\section_role_wiring_smoke` ran one CPU batch with `--model-architecture musical_plan_conditioned --section-role-plan-loss-weight 0.5 --section-role-balance-loss-weight 0.5` and logged `section_role_plan_loss 0.453` plus `section_role_balance_loss 0.180`. Old-checkpoint inference compatibility on the role+texture checkpoint reported missing `section_role_key_proj` and `section_role_head` keys and zero-initialized the compatible new modules.

Explicit arranger-state planning is now wired as the first architecture step beyond independent scalar heads. New caches write `target_arranger_state (N,8,T)` for bass rhythm, bass sustain, chord body, inner motion, top line, section bass continuity, section fullness, and section-transition emphasis; older caches derive it on load using cache `frame_hz`. `PianoRollGenerator` emits `arranger_state`, `musical_plan_conditioned` injects it with a zero-initialized projection, and `piano_roll_loss` logs `arranger_state_plan_loss`. Public cache smoke `saves2\piano_arranger\cache\arranger_state_1track_smoke` built one row and wrote `target_arranger_state_shape [8,128]`. Public CLI train smoke `saves2\piano_arranger\runs\arranger_state_wiring_smoke` ran one CUDA batch on old `bootstrap_16track_8s_planheads` with `--model-architecture musical_plan_conditioned --arranger-state-plan-loss-weight 0.1`, derived the old-cache target, and logged `arranger_state_plan_loss 0.460`. Old-checkpoint inference compatibility on the lower-weight texture-role checkpoint reported missing `arranger_state_key_proj` and `arranger_state_head`, zero-initialized those modules, and preserved loaded `texture_role_head` weights.

First arranger-state training evidence is mixed/negative, not a new best. `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_arrangerstate010_rolescore_lowlr_smoke` trained 5 CUDA epochs with lower texture-role `0.2`, role-balanced scoring, and `--arranger-state-plan-loss-weight 0.1`; `arranger_state_plan_loss` fell `0.448 -> 0.358`, but all sample eval epochs failed and best sample-eval selected epoch 1 with score `-139.495`. The controlled no-repair 24s render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_arrangerstate010_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_arrangerstate010_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` produced 87 notes, 12 unique pitches, chord-frame `0.447`, bass coverage `0.742`, melody coverage `0.041`, mean polyphony `2.770`, fullness `0.766`, source active chroma `0.386`, source onset correlation `0.556`, peak alignment `0.933`, velocity std `7.349`, and failed global eval with `harmony_collapse` plus `melody_coverage_gap`. Section report warned only `section_pitch_underuse:0,2` and had useful continuity minima: min section notes `11`, min bass `0.349`, min chord `0.150`, min fullness `0.597`. This suggests the state helps section continuity/source timing but overweights bass and underweights top-line/body; do not add direct arranger-state balance yet.

Split arranger-state planning is now wired to address that bass-heavy arranger-state failure. New caches write `target_bass_continuity (N,4,T)` for bass rhythm, bass sustain, section bass continuity, and section transitions, plus `target_body_melody_state (N,6,T)` for chord body, inner motion, top line, high activity, section body, and section melody; older caches derive both on load. `PianoRollGenerator` emits both heads, `musical_plan_conditioned` injects both with zero-initialized projections, and `piano_roll_loss` logs `bass_continuity_plan_loss` plus `body_melody_state_plan_loss`. Public cache smoke `saves2\piano_arranger\cache\split_arranger_state_1track_smoke` built one row and wrote `target_bass_continuity_shape [4,128]` plus `target_body_melody_state_shape [6,128]`. Public CLI train smoke `saves2\piano_arranger\runs\split_arranger_state_wiring_smoke` ran one CUDA batch on old `bootstrap_16track_8s_planheads` with `--bass-continuity-plan-loss-weight 0.1 --body-melody-state-plan-loss-weight 0.1`, derived both targets, and logged `bass_continuity_plan_loss 0.527` plus `body_melody_state_plan_loss 0.481`. Old-checkpoint inference compatibility zero-initialized only the missing split modules and preserved loaded `texture_role_head` weights.

Split arranger-state training is useful but not a new best. `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_splitstate010_rolescore_lowlr_smoke` trained 5 CUDA epochs with lower texture-role `0.2`, role-balanced scoring, and both split-state plan weights at `0.1`; `bass_continuity_plan_loss 0.501 -> 0.424`, `body_melody_state_plan_loss 0.427 -> 0.334`, and best sample-eval selected epoch 2 with score `-32.062`. Its controlled no-repair 24s render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_splitstate010_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_splitstate010_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` produced 138 notes, 23 unique pitches, chord-frame `0.562`, bass coverage `0.538`, melody coverage `0.077`, mean polyphony `3.859`, fullness `0.845`, source active chroma `0.447`, source onset correlation `0.507`, peak alignment `0.667`, velocity std `8.128`, and WAV RMS `0.155`. Global eval failed only `melody_coverage_gap`; section report had no warnings with min section notes `14`, min bass `0.167`, min chord `0.160`, and min fullness `0.667`. This beats the monolithic arranger-state run on diversity, chord body, section health, and harmony collapse, but still loses too much top-line and body/RMS versus the repaired lower-weight texture-role reference.

Body-melody recovery experiments are negative at the first tested settings. Raising only body-melody plan weight to `0.3` with bass-continuity `0.05` (`bootstrap_16track_8s_roletexture_melody_texturerole020_splitbody030_bass005_rolescore_lowlr_smoke`) lifted controlled-render melody coverage to `0.370`, but thinned the output to 76 notes, mean polyphony `2.621`, RMS `0.112`, and failed global eval with `harmony_collapse`; section report warned `section_pitch_underuse:2,5`. The direct `body_melody_state_balance_loss` is wired and exposed as `--body-melody-state-balance-loss-weight`; smoke `saves2\piano_arranger\runs\body_melody_state_balance_wiring_smoke` logged `body_melody_state_balance_loss 0.0898`. The first light-balance comparison (`bootstrap_16track_8s_roletexture_melody_texturerole020_splitbodybal010_bass005_rolescore_lowlr_smoke`, bass `0.05`, body plan `0.1`, body balance `0.1`) rendered 72 notes, 20 unique pitches, chord-frame `0.350`, bass `0.688`, melody `0.127`, mean polyphony `2.480`, fullness `0.809`, source onset correlation `0.433`, RMS `0.119`, and failed `harmony_collapse`; section warnings were `section_pitch_underuse:1,2` and `section_high_register_absent:3`. Keep the new balance hook, but do not reuse these weights as a candidate recipe.

Section-diversity planning is now wired for the pitch-underuse failure exposed by section-role training. New caches write `target_section_diversity (N,4,T)` for local section unique-pitch coverage, pitch-class coverage, pitch range, and onset density; older caches derive it on load using cache `frame_hz`. `PianoRollGenerator` emits `section_diversity`, `musical_plan_conditioned` injects it with a zero-initialized projection, and `piano_roll_loss` logs `section_diversity_plan_loss` plus `section_diversity_balance_loss`. The balance term is now one-sided underuse pressure, not symmetric L1. Backward-compatible dataset/model/loss smoke on old `bootstrap_16track_8s_planheads` derived `section_diversity (1,4,256)`, predicted `section_diversity (1,4,256)`, and produced `section_diversity_plan_loss 0.308` plus `section_diversity_balance_loss 0.263` before that refinement. Public CLI smoke `saves2\piano_arranger\runs\section_diversity_wiring_smoke` ran one CPU batch with `--model-architecture musical_plan_conditioned --section-diversity-plan-loss-weight 0.1 --section-diversity-balance-loss-weight 0.1` and logged `section_diversity_plan_loss 0.225` plus `section_diversity_balance_loss 0.473`. After the one-sided underuse refinement, `saves2\piano_arranger\runs\section_diversity_underuse_wiring_smoke` compiled and ran one CUDA batch, logging `section_diversity_plan_loss 0.225` and `section_diversity_balance_loss 0.000` on that batch. Old-checkpoint inference compatibility on the lower-weight texture-role checkpoint reported missing `section_diversity_key_proj` and `section_diversity_head` keys and zero-initialized the compatible new modules.

Section-diversity training is negative evidence, not a new best. Symmetric balance run `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole002_diversity010_rolescore_lowlr_smoke` trained 5 CUDA epochs with section-role `0.02` and section-diversity `0.1`; best sample-eval score was `-133.510`. Its controlled no-repair 24s render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole002_diversity010_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_sectionrole002_diversity010_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` passed loose global eval but was far too thin: 56 notes, 11 unique pitches, chord-frame `0.130`, bass coverage `0.573`, melody coverage `0.156`, mean polyphony `1.691`, fullness `0.668`, and source onset correlation `0.441`. Section report warned `section_high_register_absent:0`, `section_pitch_underuse:1,2`, and `section_high_register_absent:3`.

One-sided underuse section-diversity did not fix the branch. `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole005_diversity010_underuse_rolescore_lowlr_smoke` trained 5 CUDA epochs with section-role `0.05`; best sample-eval score was `-107.902`. Its controlled no-repair 24s render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole005_diversity010_underuse_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_sectionrole005_diversity010_underuse_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` produced 55 notes, 11 unique pitches, chord-frame `0.130`, bass coverage `0.492`, melody coverage `0.156`, mean polyphony `1.644`, fullness `0.666`, source onset correlation `0.439`, min section notes `2`, min section bass `0.000`, min section chord `0.000`, and the same section high-register/pitch-underuse warnings. Do not keep increasing section-diversity weights; move to a richer section/role arranger state instead.

Section-role training evidence is mixed, not a new best. `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole020_rolescore_lowlr_smoke` trained 5 CUDA epochs with section-role weights `0.2`; best sample-eval epoch 4 scored `575.525` with `section_role_plan_loss 0.336` and one `single_pitch_collapse` sample warning. Its controlled 24s no-repair render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole020_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_sectionrole020_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` passed global eval with no warnings and no bassless sections, but section report warned `section_pitch_underuse` in sections 1-4. Metrics: 119 notes, 12 unique pitches, chord-frame `0.823`, bass coverage `0.484`, melody coverage `0.937`, source onset correlation `0.028`, RMS `0.201`, min section bass `0.390`, and min section fullness `0.909`.

Lower section-role weight `0.05` is a better but still incomplete variant. `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole005_rolescore_lowlr_smoke` trained 5 CUDA epochs; best sample-eval epoch 4 scored `575.533` with `section_role_plan_loss 0.356` and one `single_pitch_collapse` sample warning. Its controlled 24s no-repair render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_sectionrole005_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_sectionrole005_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` passed global eval with no warnings and improved learned section bass without repair, but section report still warned `section_pitch_underuse` in sections 0, 1, 3, and 4. Metrics: 123 notes, 12 unique pitches, chord-frame `0.898`, bass coverage `0.596`, melody coverage `0.931`, source active chroma `0.477`, source onset correlation `0.054`, RMS `0.197`, min section notes `12`, min section bass `0.120`, min section chord `0.690`, and min section fullness `0.869`. This proves the learned target can remove the bassless-section failure without `--section-bass-repair`, but it loses the repaired baseline's note diversity, density, velocity spread, and source onset tracking.

Section-aware sample scoring is now wired to catch that class of failure earlier. Smoke `saves2\piano_arranger\runs\section_sample_score_wiring_smoke` ran one CPU batch and wrote section aggregate keys into history, including `sample_eval_mean_min_section_notes`, `sample_eval_mean_min_section_bass_coverage_fraction`, `sample_eval_mean_min_section_chord_frame_fraction`, and `sample_eval_mean_min_section_fullness_score`.

Joint quality-penalty sample scoring is now wired so best-checkpoint selection penalizes thin, pitch-class-dominant, or section-weak samples even when role-balance is high. `_sample_eval_score_components` logs `quality_penalty`, `density_penalty`, `rms_penalty`, `pitch_class_penalty`, `section_notes_penalty`, `section_unique_penalty`, `section_chord_penalty`, and `section_fullness_penalty`. Smoke `saves2\piano_arranger\runs\sample_quality_selector_wiring_smoke` ran one CUDA batch/sample and logged `quality_penalty 3.75`, `section_chord_penalty 0.15`, mean single pitch-class fraction `0.2125`, and `sample_eval_score -32.931`. Section-unique selector smoke `saves2\piano_arranger\runs\section_unique_selector_wiring_smoke` ran one CPU batch with `--sample-score-min-section-unique-pitches 8`; its weak sample logged `sample_eval_mean_min_section_unique_pitches=1.0`, `section_unique_penalty=0.875`, and `sample_eval_score=-231.719`. This is selector evidence, not a musical pass.

Short section-unique selector comparison is negative but informative. `saves2\piano_arranger\runs\section_unique_selector_3epoch_smoke` trained 3 CPU epochs on `bootstrap_16track_8s_planheads`; loss fell `4.7669 -> 4.4321`, `sample_eval_mean_min_section_notes` improved `5.5 -> 9.0`, and `sample_eval_mean_min_section_unique_pitches` improved `2.0 -> 3.0`, but all samples still failed and warning count rose `13 -> 19`. Best-sample selection correctly chose epoch 1 (`sample_eval_score=-218.738`) over later lower-loss epochs. Do not use this checkpoint as an arbitrary-source candidate; the evidence says scalar selector pressure alone is not enough from a weak short run.

Quality-penalty selector evidence is useful but not sufficient. `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_splitstate010_qualityscore_rolescore_lowlr_smoke` repeated the split-state recipe with stricter selector settings. It still selected epoch 2 with score `-32.062`, but heavily penalized weaker alternatives: epoch 1 had `quality_penalty 79.76`, epoch 5 had `42.37`, and epochs 2/3 had `0.0`. The controlled no-repair 24s render `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_splitstate010_qualityscore_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_densitymatch_pitchcap18_splitstate010_qualityscore_no_bassrepair_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` produced 135 notes, 22 unique pitches, chord-frame `0.542`, bass `0.538`, melody `0.077`, mean polyphony `3.760`, fullness `0.840`, source active chroma `0.446`, source onset correlation `0.505`, RMS `0.158`, and failed global eval only with `melody_coverage_gap`; section report had no warnings. The selector is a guardrail for long runs, not a solution to the top-line/body tradeoff.

Joint texture-role training result: `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerolemix_lowlr_smoke` trained 5 CUDA epochs with role, texture, melody, and texture-role weights all at `0.5`; `texture_role_plan_loss 0.383 -> 0.294`, while normalized `texture_role_balance_loss` stayed near `0.240`. Sample eval still selected epoch 1 because later epochs accumulated `velocity_flat` and `mid_harmony_underuse` warnings. The artifact `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerolemix_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roletexture_melody_texturerolemix_velboost_model_piano.{json,mid,wav}` passed global and section eval with 182 notes, 21 unique pitches, bass coverage `0.901`, melody coverage `0.152`, chord-frame fraction `0.715`, mean polyphony `5.643`, source active chroma `0.405`, source onset correlation `0.554`, source peak alignment `0.875`, fullness `0.901`, and RMS `0.201`. This is useful negative evidence: equal-weight joint texture-role pressure improves neither the role+texture body nor the melody-aware top line enough to become the new best.

Role-balanced lower-weight texture-role result: `saves2\piano_arranger\runs\bootstrap_16track_8s_roletexture_melody_texturerole020_rolescore_lowlr_smoke` trained 5 CUDA epochs with texture-role weights at `0.2`, role-balanced sample scoring (`--sample-score-role-balance-weight 100`), and selected epoch 2. Sample scoring chose it because role-balance was strong despite one `velocity_flat` warning: sample chord-frame `0.951`, melody coverage `0.572`, bass coverage `0.802`, mean polyphony `8.450`, fullness `1.000`, and RMS `0.270`. The first 24s decode with pitch cap `0.20` failed only by `single_pitch_collapse` (`single_pitch_fraction 0.2509`). The stricter pitch-cap artifact `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap18_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}` passed global eval with 267 notes, 25 unique pitches, chord-frame `0.797`, bass coverage `0.434`, melody coverage `0.241`, mean polyphony `6.008`, fullness `0.961`, source active chroma `0.416`, source onset correlation `0.597`, source peak alignment `0.808`, velocity std `7.341`, and RMS `0.216`, but section report still warned `section_bass_absent:0`.

Section bass repair result: re-rendering that checkpoint with pitch cap `0.18`, `--section-bass-repair`, and `--section-bass-repair-min-coverage 0.05` produced `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_texturerole020_rolescore_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassrepair_pitchcap18_roletexture_melody_texturerole020_rolescore_velboost_model_piano.{json,mid,wav}`. Decode metadata reports one section-bass repair in chunk 0. Global and section eval both pass with no warnings: 268 notes, 25 unique pitches, chord-frame `0.810`, bass coverage `0.451`, melody coverage `0.241`, mean polyphony `6.025`, fullness `0.961`, source active chroma `0.419`, source onset correlation `0.601`, source peak alignment `0.810`, velocity std `7.328`, and RMS `0.216`. Section minimum bass coverage is now `0.102`, minimum chord-frame `0.410`, and minimum fullness `0.696`. This is the strongest mixed body/melody/onset pseudo-target artifact so far, but the bass continuity gain comes from decode repair rather than learned section planning.

General-source chunked artifact: `saves2\piano_arranger\outputs\bootstrap_16track_8s_musicalplan_eventdist_lowlr_smoke\hui_deewani_8s_chunked_model_piano.{json,mid}` used the low-LR checkpoint on `Hui Deewani Krishan Ki He Meera` from the broad source manifest. It produced 52 notes over 8 seconds across two 4-second chunks; source-aware eval passed with 13 unique pitches, max simultaneity `10`, single pitch-class fraction `0.308`, source active chroma `0.379`, source onset correlation `0.687`, and source peak alignment `0.864`. Section reporting now folds tiny trailing duration fragments into the previous section; the section report has 2 sections, no warnings, min section notes `23`, and min high-register fraction `0.435`. This is broader pseudo-pretraining and source-following evidence, not supervised paired quality evidence.

Fullness-aware eval now reports role coverage: bass/mid/high note fractions, weighted mean velocity, chord-frame coverage, bass coverage, melody coverage, mean active polyphony, and a bounded fullness score. Section reports also warn on `section_bass_absent` and `section_thin_texture`. The earlier longer artifact `hui_deewani_24s_chunked_model_piano.{json,mid,wav}` still passed global source-aware eval, but the new section report exposed `section_bass_absent:1`, so it is no longer the best fullness evidence.

Current best longer general-source chunked artifact: `saves2\piano_arranger\outputs\bootstrap_16track_8s_musicalplan_eventdist_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_model_piano.{json,mid,wav}` used the same low-LR checkpoint for 24 seconds with 4-second chunks, a 2-second hop, `--section-profile arc`, `--bass-min-note-duration 0.35`, and `--max-pitch-fraction 0.20`. Source-aware eval passed with no warnings: 144 notes, 17 unique pitches, pitch range `87`, max simultaneity `11`, single pitch fraction `0.243`, single pitch-class fraction `0.257`, source active chroma `0.404`, source onset correlation `0.598`, source peak alignment `0.841`, global bass coverage `0.369`, chord-frame fraction `0.807`, and fullness score `1.000`. Section report also passed with no warnings across 6 sections: min section notes `15`, min bass coverage `0.100`, min chord-frame fraction `0.530`, and min fullness score `0.787`. The WAV sanity check passed at 24.6s, 22.05 kHz, peak `0.950`, RMS `0.162`. This is the current best longer-form source-following artifact for the pseudo-pretrained path, but it is still pseudo-target evidence, not real paired target proof.

Current role/fullness-weighted artifact: `saves2\piano_arranger\outputs\bootstrap_16track_8s_roleplan_neutral_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roleplan_neutral_velboost_model_piano.{json,mid,wav}` passed the same global and section gates with higher source chroma, bass coverage, velocity variation, and RMS than the non-role artifact, but only 111 notes. Treat it as useful architecture evidence, not a replacement for the denser reference. The next model step should keep the role-plan RMS/bass gains while improving mid-register harmony/body and note density.

Current best fullness/body artifact: `saves2\piano_arranger\outputs\bootstrap_16track_8s_roleplan_texture_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roletexture_velboost_model_piano.{json,mid,wav}` is the strongest current artifact for the "full sounding piano" direction: it is denser than the non-role reference and stronger in RMS/mid-body than the role-only run. Treat its lower melody/high-register coverage as the next pressure point, not as solved final quality.

Current melody/top-line artifact: `saves2\piano_arranger\outputs\bootstrap_16track_8s_roletexture_melody_lowlr_smoke\hui_deewani_24s_chunked_hop2_bassfloor_pitchcap20_roletexture_melody_velboost_model_piano.{json,mid,wav}` proves the new melody target can lift top-line coverage, but it is not the new best fullness artifact because it loses chord-frame density and RMS. The next model step should make melody, bass, chord body, and inner rhythm explicit enough to optimize jointly instead of shifting weight from one role to another.

Next training hypothesis: do not repeat equal `0.5` texture-role weighting, do not blindly increase `section_role` weight, and do not continue section-diversity pressure as the main solution. The lower `0.2` texture-role recipe remains useful, but `section_role` weights `0.2` and `0.05` overcorrect toward narrow pitch use while improving bass continuity; section-diversity variants at `0.1` make the controlled render thin instead of full. Split bass-continuity/body-melody plus quality-aware checkpoint selection is better than monolithic arranger-state for section health, but melody coverage remains unsolved. The next model move should not be another scalar head-weight sweep; move to real paired/full-song target evidence or change the generator/checkpoint selector so body, top line, source timing, pitch-class diversity, and section fullness are optimized jointly.

## Baseline Controls

The heuristic baseline is not the final model. It exists to define inspectable arrangement artifacts before training.

Useful controls:

```powershell
python "lab 3.6\run_piano_arranger_pipeline.py" --action heuristic-baseline `
  --source-audio "<song.wav>" `
  --seconds 30 `
  --fullness 0.9 `
  --melody-focus 0.8 `
  --rhythmic-drive 0.7 `
  --harmonic-adventure 0.3 `
  --register-width 0.9 `
  --pedal-amount 0.7
```

## Non-Goal

Do not treat audio timbre transfer as sufficient. A valid Lab 3.6 result must expose arrangement structure, preferably MIDI plus WAV.
