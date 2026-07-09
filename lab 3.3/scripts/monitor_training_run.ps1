param(
    [Parameter(Mandatory=$true)][int]$TrainPid,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [Parameter(Mandatory=$true)][string]$StdoutLog,
    [Parameter(Mandatory=$true)][string]$StderrLog,
    [Parameter(Mandatory=$true)][string]$MonitorLog,
    [int]$IntervalSec = 1500
)

$ErrorActionPreference = "SilentlyContinue"

function Get-LatestProgressLine([string]$Path) {
    if (-not (Test-Path $Path)) { return "" }
    $lines = Get-Content $Path -Tail 120
    $matches = $lines | Where-Object { $_ -match '^\[epoch ' -or $_ -match '^epoch=' -or $_ -match '^resumed_from=' }
    if ($matches) { return ($matches | Select-Object -Last 1) }
    return ""
}

function Get-LatestErrLine([string]$Path) {
    if (-not (Test-Path $Path)) { return "" }
    $lines = Get-Content $Path -Tail 80
    $nonEmpty = $lines | Where-Object { $_.Trim().Length -gt 0 }
    if ($nonEmpty) { return ($nonEmpty | Select-Object -Last 1) }
    return ""
}

function Get-EpochCount([string]$Path) {
    $history = Join-Path $Path "v2_history.json"
    if (-not (Test-Path $history)) { return 0 }
    try {
        $obj = Get-Content $history -Raw | ConvertFrom-Json
        if ($obj -is [System.Array]) { return $obj.Count }
        return 0
    } catch {
        return 0
    }
}

function Get-SampleWavCount([string]$Path) {
    $samples = Join-Path $Path "epoch_samples"
    if (-not (Test-Path $samples)) { return 0 }
    return @(Get-ChildItem $samples -Recurse -Filter *.wav).Count
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $MonitorLog) | Out-Null

while ($true) {
    $alive = [bool](Get-Process -Id $TrainPid)
    $payload = [ordered]@{
        ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        alive = $alive
        epochs_done = (Get-EpochCount $RunDir)
        sample_wavs = (Get-SampleWavCount $RunDir)
        summary_exists = (Test-Path (Join-Path $RunDir "summary.json"))
        progress = (Get-LatestProgressLine $StdoutLog)
        stderr = (Get-LatestErrLine $StderrLog)
    } | ConvertTo-Json -Compress

    Add-Content -Path $MonitorLog -Value $payload
    if (-not $alive) { break }
    Start-Sleep -Seconds ([Math]::Max(30, $IntervalSec))
}
