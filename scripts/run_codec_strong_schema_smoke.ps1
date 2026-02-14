param(
  [string]$OutRoot = "saves2/lab3_codec_transfer",
  [int]$Seed = 328
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Fast preflight: validates ingestion schema, cache build, style-judge, style-bank, and one tiny training step.
python "lab 3/run_lab3_codec.py" `
  --mode fresh `
  --out-root $OutRoot `
  --seed $Seed `
  --genre-schema binary_acoustic_beats `
  --require-min-sources-per-genre 2 `
  --balance-sources-within-genre `
  --require-is-music `
  --refit-style-judge `
  --fail-on-style-judge-weak `
  --fail-on-style-bank-collapse `
  --style-cond-source codec_judge_embed `
  --style-loss-mode codec_judge_ce `
  --style-judge-mode codec_judge `
  --style-judge-epochs 8 `
  --style-judge-min-val-acc 0.75 `
  --per-genre-samples 200 `
  --stage1-epochs 1 `
  --stage2-epochs 1 `
  --stage3-epochs 1 `
  --max-batches-per-epoch 4 `
  --auto-sample-export `
  --sample-count 8 `
  --smoke

