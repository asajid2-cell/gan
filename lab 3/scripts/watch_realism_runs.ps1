$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$paths = @(
    "saves2/lab3_codec_transfer/run1056/logs/train_codec_realism.log",
    "saves2/lab3_codec_transfer/run1056/logs/realism_sweep.log",
    "saves2/lab3_diffusion/run_d004/logs/train_diffusion_realism.log",
    "saves2/lab3_diffusion/run_d004/logs/realism_sweep.log",
    "saves2/_supervision/diffusion_overnight_run_d004.stdout.log",
    "saves2/_supervision/diffusion_overnight_run_d004.stderr.log"
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host ""
        Write-Host "===== $path ====="
        Get-Content $path -Tail 40
    }
}

Write-Host ""
Write-Host "===== checkpoint snapshots ====="
if (Test-Path "saves2/lab3_codec_transfer/run1056/checkpoints") {
    Get-ChildItem "saves2/lab3_codec_transfer/run1056/checkpoints" |
        Select-Object Name, LastWriteTime, Length |
        Format-Table -AutoSize
}
if (Test-Path "saves2/lab3_diffusion/run_d004/checkpoints") {
    Get-ChildItem "saves2/lab3_diffusion/run_d004/checkpoints" |
        Select-Object Name, LastWriteTime, Length |
        Format-Table -AutoSize
}
