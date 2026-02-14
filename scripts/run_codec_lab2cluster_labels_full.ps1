param(
  [string]$OutRoot = "saves2/lab3_codec_transfer",
  [int]$K = 4,
  [int]$Seed = 328,
  [int]$MaxFiles = 0,
  [int]$PerGenreSamples = 800,
  [int]$MinClusterSize = 100,
  [ValidateSet("abort","unassigned","merge")]
  [string]$SmallClusterAction = "merge",
  [switch]$FailOnStyleBankCollapse = $false,
  [double]$StyleBankMinNearestCentroidAcc = 0.55
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$manifestDir = "saves2"
$manifestName = "auto_cluster_k${K}_${ts}.csv"
$manifestPath = Join-Path $manifestDir $manifestName

Write-Host ("[lab2cluster] labeling -> {0}" -f $manifestPath)
python "lab 3/run_lab3_auto_genre_lab2cluster.py" `
  --out-csv $manifestPath `
  --n-clusters $K `
  --seed $Seed `
  --max-files $MaxFiles `
  --min-conf 0.0 `
  --min-cluster-size $MinClusterSize `
  --small-cluster-action $SmallClusterAction `
  --require-min-sources-per-cluster 1 `
  --max-source-fraction-per-cluster 1.0

Write-Host ("[codec] training with manifest {0}" -f $manifestName)
$styleBankFailFlag = "--no-fail-on-style-bank-collapse"
if ($FailOnStyleBankCollapse) { $styleBankFailFlag = "--fail-on-style-bank-collapse" }

python "lab 3/run_lab3_codec.py" `
  --mode fresh `
  --out-root $OutRoot `
  --seed $Seed `
  --manifests-root (Resolve-Path $manifestDir) `
  --manifest-files $manifestName `
  --genre-schema default4 `
  --require-min-sources-per-genre 1 `
  --no-balance-sources-within-genre `
  --per-genre-samples $PerGenreSamples `
  --refit-style-judge `
  --style-judge-mode codec_judge `
  --style-loss-mode codec_judge_ce `
  --style-cond-source codec_judge_embed `
  --style-bank-min-nearest-centroid-acc $StyleBankMinNearestCentroidAcc `
  $styleBankFailFlag `
  --auto-sample-export

Write-Host "[audit] running leakage audit on latest run"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_codec_audit_latest.ps1 -OutRoot $OutRoot -Seed $Seed
