param([string]$RunName = "run_d004")

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$supervisionDir = Join-Path $repoRoot "saves2\_supervision"
New-Item -ItemType Directory -Force -Path $supervisionDir | Out-Null

$stdoutLog = Join-Path $supervisionDir ("diffusion_overnight_" + $RunName + ".stdout.log")
$stderrLog = Join-Path $supervisionDir ("diffusion_overnight_" + $RunName + ".stderr.log")
$script = (Resolve-Path (Join-Path $PSScriptRoot "start_diffusion_realism_run.ps1")).Path

if (Test-Path (Join-Path $repoRoot ("saves2\lab3_diffusion\" + $RunName))) {
    throw "Refusing to overwrite existing diffusion run: $RunName"
}

$proc = Start-Process powershell `
    -ArgumentList @('-ExecutionPolicy','Bypass','-Command', "& '$script' -RunName '$RunName'") `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Write-Host ("Started overnight diffusion run " + $RunName + " with PID " + $proc.Id)
Write-Host ("stdout: " + $stdoutLog)
Write-Host ("stderr: " + $stderrLog)
