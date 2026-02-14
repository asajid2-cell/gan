param(
  [string]$OutRoot = "saves2/lab3_codec_transfer",
  [int]$Seed = 328,
  [string]$PromptTemplate = "a {label} music track",
  [double]$MinConf = 0.25,
  [int]$MaxFiles = 0,
  [int]$PerGenreSamples = 600,
  [int]$RequireMinSourcesPerGenre = 2,
  [switch]$BalanceSourcesWithinGenre = $true,
  [switch]$RequireIsMusic = $false,
  [int]$StyleJudgeEpochs = 12,
  [int]$BatchSize = 2,
  [int]$TranslatorHiddenChannels = 192,
  [int]$TranslatorBlocks = 8,
  [int]$DiscriminatorScales = 2,
  [int]$Stage1Epochs = 1,
  [int]$Stage2Epochs = 12,
  [int]$Stage3Epochs = 8,
  [int]$MaxBatchesPerEpoch = 36,
  [double]$Stage2ContentWeight = 2.0,
  [double]$Stage2StyleWeight = 11.0,
  [double]$Stage3ContentWeight = 1.6,
  [double]$Stage3StyleWeight = 14.0,
  [double]$Stage2StylePushWeight = 1.5,
  [double]$Stage3StylePushWeight = 2.0,
  [double]$Stage2StyleEmbedAlignWeight = 0.8,
  [double]$Stage3StyleEmbedAlignWeight = 1.0,
  [double]$Stage3ModeSeekingWeight = 0.08,
  [double]$StyleBankMinNearestCentroidAcc = 0.55,
  [switch]$FailOnStyleBankCollapse = $false,
  [string[]]$Labels = @("classical", "hiphop", "electronic", "ambient")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Keep label set small/coarse so CLAP confidence is usable for filtering.
if (-not $Labels -or $Labels.Count -lt 2) {
  throw "Need at least 2 labels via -Labels."
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$manifestDir = "saves2"
$manifestName = "auto_clap_${ts}.csv"
$manifestPath = Join-Path $manifestDir $manifestName

Write-Host ("[clap] labeling -> {0}" -f $manifestPath)
python "lab 3/run_lab3_auto_genre.py" `
  --out-csv $manifestPath `
  --labels $Labels `
  --prompt-template $PromptTemplate `
  --min-conf $MinConf `
  --seed $Seed `
  --max-files $MaxFiles

$balanceFlag = "--no-balance-sources-within-genre"
if ($BalanceSourcesWithinGenre) { $balanceFlag = "--balance-sources-within-genre" }

$musicFlag = "--no-require-is-music"
if ($RequireIsMusic) { $musicFlag = "--require-is-music" }

$styleBankFailFlag = "--no-fail-on-style-bank-collapse"
if ($FailOnStyleBankCollapse) { $styleBankFailFlag = "--fail-on-style-bank-collapse" }

Write-Host ("[codec] training with CLAP-labeled manifest {0}" -f $manifestName)
python "lab 3/run_lab3_codec.py" `
  --mode fresh `
  --out-root $OutRoot `
  --seed $Seed `
  --manifests-root (Resolve-Path $manifestDir) `
  --manifest-files $manifestName `
  --genre-schema default4 `
  --require-min-sources-per-genre $RequireMinSourcesPerGenre `
  $balanceFlag `
  $musicFlag `
  --per-genre-samples $PerGenreSamples `
  --batch-size $BatchSize `
  --translator-hidden-channels $TranslatorHiddenChannels `
  --translator-blocks $TranslatorBlocks `
  --discriminator-scales $DiscriminatorScales `
  --stage1-epochs $Stage1Epochs `
  --stage2-epochs $Stage2Epochs `
  --stage3-epochs $Stage3Epochs `
  --max-batches-per-epoch $MaxBatchesPerEpoch `
  --stage2-content-weight $Stage2ContentWeight `
  --stage2-style-weight $Stage2StyleWeight `
  --stage3-content-weight $Stage3ContentWeight `
  --stage3-style-weight $Stage3StyleWeight `
  --stage2-style-push-weight $Stage2StylePushWeight `
  --stage3-style-push-weight $Stage3StylePushWeight `
  --stage2-style-embed-align-weight $Stage2StyleEmbedAlignWeight `
  --stage3-style-embed-align-weight $Stage3StyleEmbedAlignWeight `
  --stage3-mode-seeking-weight $Stage3ModeSeekingWeight `
  --refit-style-judge `
  --style-judge-mode codec_judge `
  --style-loss-mode codec_judge_ce `
  --style-cond-source codec_judge_embed `
  --style-judge-epochs $StyleJudgeEpochs `
  --style-bank-min-nearest-centroid-acc $StyleBankMinNearestCentroidAcc `
  $styleBankFailFlag `
  --auto-sample-export

Write-Host "[audit] running leakage audit on latest run"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_codec_audit_latest.ps1 -OutRoot $OutRoot -Seed $Seed
