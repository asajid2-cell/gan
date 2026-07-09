# Lab 3.5 Real-Music Transfer Pipeline

This folder contains the real-music reset path for the genre-transfer work.
It does not assume the old Lab 1/2 curated manifests. It can ingest raw audio
folders, discover style families from metadata/audio characteristics, build a
diffusion feature cache, train a retrieval-conditioned generator, and run
audio-to-audio inference.

Default dataset:

```powershell
python "lab 3.5/run_real_music_pipeline.py" --action manifest
```

The default points at:

```text
Z:\DataSets\annas_archive_data__aacid__spotify_files_pop_0__20260116T000346Z--20260116T000347Z
```

Discovered-style manifest:

```powershell
python "lab 3.5/run_real_music_pipeline.py" `
  --action discover `
  --default-data-root "Z:\DataSets\annas_archive_data__aacid__spotify_files_pop_0__20260116T000346Z--20260116T000347Z" `
  --discover-manifest-path "data\real_music_manifests\spotify_discovered_genres.csv" `
  --discovery-report-path "data\real_music_manifests\spotify_discovered_genres_report.json" `
  --discover-clusters 16 `
  --discover-audio-feature-limit 0 `
  --discover-audio-workers 6
```

Production cache build:

```powershell
python "lab 3.5/run_real_music_pipeline.py" `
  --action cache `
  --manifest-path "data\real_music_manifests\spotify_discovered_genres.csv" `
  --cache-dir "saves2\real_music_transfer\spotify_discovered_genres_cache" `
  --max-chunks-per-track 4 `
  --progress-every 200 `
  --shard-size 5000
```

Production training:

```powershell
python "lab 3.5/run_real_music_pipeline.py" `
  --action train `
  --cache-dir "saves2\real_music_transfer\spotify_discovered_genres_cache" `
  --resume-checkpoint "saves2\real_music_transfer\runs\real_transfer_20260512_182741\checkpoints\partial.pt" `
  --resume-out-dir "saves2\real_music_transfer\runs\real_transfer_20260512_182741" `
  --epochs 8 `
  --batch-size 4 `
  --checkpoint-every-batches 1000
```

Omit the resume flags for a fresh run. The completed production run is:

```text
saves2\real_music_transfer\runs\real_transfer_20260512_182741
```

It contains `summary.json`, `history.json`, `checkpoints\latest.pt`,
`checkpoints\best_by_val.pt`, and `epoch_001.pt` through `epoch_008.pt`.

Inference against a discovered style family:

```powershell
python "lab 3.5/run_real_music_pipeline.py" `
  --action infer `
  --cache-dir "saves2\real_music_transfer\spotify_discovered_genres_cache" `
  --checkpoint "saves2\real_music_transfer\runs\real_transfer_20260512_182741\checkpoints\best_by_val.pt" `
  --source-audio "<source.wav-or-ogg>" `
  --target-genre "style_07_fast_bright_percussive_loud_die_kapitel_magie" `
  --out-wav "saves2\real_music_transfer\outputs\source_to_style_07.wav" `
  --infer-seconds 24
```

Validation and reports:

```powershell
python "lab 3.5/run_real_music_validation.py" --action evaluate `
  --cache-dir "saves2\real_music_transfer\spotify_discovered_genres_cache" `
  --plan-path "saves2\real_music_transfer\validation_plan.json" `
  --pack-dir "saves2\real_music_transfer\final_pack_strength3" `
  --report-path "saves2\real_music_transfer\final_pack_strength3\validation_report.json" `
  --profiles-path "saves2\real_music_transfer\reference_profiles.json"

python "lab 3.5/run_real_music_reports.py" --action all `
  --validation-pack-dir "saves2\real_music_transfer\final_pack_strength3" `
  --validation-report "saves2\real_music_transfer\final_pack_strength3\validation_report.json" `
  --final-pack-dir "saves2\real_music_transfer\delivery_pack_strength3" `
  --separation-report "saves2\real_music_transfer\genre_separation_report_strength3.json" `
  --listening-audit "saves2\real_music_transfer\listening_audit_strength3.json" `
  --baseline-report "saves2\real_music_transfer\baseline_compare_report_strength3.json" `
  --discovery-report "data\real_music_manifests\spotify_discovered_genres_report.json" `
  --cache-dir "saves2\real_music_transfer\spotify_discovered_genres_cache" `
  --train-summary "saves2\real_music_transfer\runs\real_transfer_20260512_182741\summary.json" `
  --validation-plan "saves2\real_music_transfer\validation_plan.json" `
  --gate-report "saves2\real_music_transfer\completion_gate_report_strength3.json"
```

Monitoring for long GPU jobs:

```powershell
nvidia-smi
Get-CimInstance Win32_OperatingSystem
Get-Process python | Select-Object Id,CPU,@{n='MemGB';e={[math]::Round($_.WorkingSet64/1GB,2)}}
Get-Content "saves2\real_music_transfer\logs\<job>_stdout.log" -Tail 20
Get-Content "saves2\real_music_transfer\logs\<job>_stderr.log" -Tail 20
```

Stop and resume from `checkpoints\partial.pt` if free RAM falls near the safety
floor. Do not mark the goal complete from training artifacts alone; the current
completion gate still requires same-clip baseline comparison and manual
listening notes.
