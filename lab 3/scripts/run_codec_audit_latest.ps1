param(
  [string]$OutRoot = "saves2/lab3_codec_transfer",
  [int]$Seed = 328
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Resolve-Path $OutRoot
$latest = Get-ChildItem $root -Directory |
  Where-Object { $_.Name -match '^run\d+$' } |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

if (-not $latest) {
  throw "No run dirs found under $root"
}

Write-Host ("[audit] latest run: {0}" -f $latest.FullName)
python "lab 3/run_lab3_target_vector_audit.py" --run-dir $latest.FullName --seed $Seed
