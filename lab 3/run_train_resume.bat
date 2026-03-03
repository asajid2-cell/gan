@echo off
echo ===== Resuming Diffusion Training from Checkpoint =====
set LAB3_DIR=%~dp0
set REPO_ROOT=%LAB3_DIR%..
echo Repo root: %REPO_ROOT%
echo Output dir: %REPO_ROOT%\saves2\lab3_diffusion\run_d002
echo.

cd /d "%LAB3_DIR%"

python run_lab3_diffusion_v2.py ^
    --cache-dir "%REPO_ROOT%\saves2\lab3_diffusion\run_d001\cache" ^
    --out-dir "%REPO_ROOT%\saves2\lab3_diffusion\run_d002" ^
    --batch-size 4 ^
    --log-every 50 ^
    --num-workers 0 ^
    --epochs 60

echo.
echo ===== Training Complete =====
pause
