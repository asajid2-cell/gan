param(
  [string]$OutRoot = "saves2/lab3_codec_transfer",
  [int]$Seed = 328
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Full run tuned for "unpaired validity":
# - schema merges into multi-source buckets (reduces source/domain proxying)
# - balances sources within each genre bucket
# - filters to music-only clips when manifests provide is_music
# - requires a competent codec-native style judge and non-collapsed style bank
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
  --style-judge-epochs 20 `
  --style-judge-min-val-acc 0.80 `
  --per-genre-samples 800 `
  --stage1-epochs 6 `
  --stage2-epochs 20 `
  --stage3-epochs 16 `
  --auto-sample-export `
  --sample-count 24

