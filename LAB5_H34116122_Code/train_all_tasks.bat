@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Spring 2026 DLP Lab5 / HW1 - Train Task 1, Task 2, Task 3
REM Put this .bat inside: LAB5_H34116122_Code
REM It will activate the parent virtual environment: ..\.venv
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo Current directory:
cd
echo ============================================================

REM ---------- Activate virtual environment ----------
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [ERROR] Cannot find virtual environment activate.bat.
    echo Expected: ..\.venv\Scripts\activate.bat
    echo Please activate venv manually or place this file correctly.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Python environment check
echo ============================================================
python --version
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 (
    echo [ERROR] Python / PyTorch environment check failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Task 1: Vanilla DQN on CartPole-v1
echo ============================================================
python dqn.py ^
  --episodes 800 ^
  --save-dir ./results_task1 ^
  --wandb-run-name task1-cartpole-dqn ^
  --batch-size 64 ^
  --memory-size 50000 ^
  --lr 0.001 ^
  --discount-factor 0.99 ^
  --epsilon-start 1.0 ^
  --epsilon-decay 0.995 ^
  --epsilon-min 0.05 ^
  --target-update-frequency 500 ^
  --replay-start-size 1000 ^
  --max-episode-steps 500 ^
  --train-per-step 1

if errorlevel 1 (
    echo [ERROR] Task 1 training failed. Stop.
    pause
    exit /b 1
)

if exist ".\results_task1\best_model.pt" (
    copy /Y ".\results_task1\best_model.pt" "..\LAB5_H34116122_task1.pt"
    echo [OK] Copied Task 1 model to ..\LAB5_H34116122_task1.pt
) else (
    echo [WARN] Task 1 best_model.pt not found.
)

echo.
echo ============================================================
echo Task 2: Vanilla DQN on Pong-v5
echo ============================================================
python dqn_task2.py ^
  --episodes 5000 ^
  --save-dir ./results_task2 ^
  --wandb-run-name task2-pong-vanilla-dqn ^
  --batch-size 32 ^
  --memory-size 100000 ^
  --lr 0.0001 ^
  --discount-factor 0.99 ^
  --epsilon-start 1.0 ^
  --epsilon-decay 0.999995 ^
  --epsilon-min 0.05 ^
  --target-update-frequency 1000 ^
  --replay-start-size 50000 ^
  --max-episode-steps 10000 ^
  --train-per-step 1

if errorlevel 1 (
    echo [ERROR] Task 2 training failed. Stop.
    pause
    exit /b 1
)

if exist ".\results_task2\best_model.pt" (
    copy /Y ".\results_task2\best_model.pt" "..\LAB5_H34116122_task2.pt"
    echo [OK] Copied Task 2 model to ..\LAB5_H34116122_task2.pt
) else (
    echo [WARN] Task 2 best_model.pt not found.
)

echo.
echo ============================================================
echo Task 3: Enhanced DQN on Pong-v5
echo Enhancements: Double DQN + PER + Multi-Step Return
echo ============================================================
python dqn_task3.py ^
  --episodes 20000 ^
  --save-dir ./results_task3 ^
  --wandb-run-name task3-pong-enhanced-dqn ^
  --batch-size 32 ^
  --memory-size 100000 ^
  --lr 0.0001 ^
  --discount-factor 0.99 ^
  --epsilon-start 1.0 ^
  --epsilon-decay 0.999995 ^
  --epsilon-min 0.05 ^
  --target-update-frequency 1000 ^
  --replay-start-size 50000 ^
  --max-episode-steps 10000 ^
  --train-per-step 1 ^
  --n-step 3 ^
  --per-alpha 0.6 ^
  --per-beta 0.4

if errorlevel 1 (
    echo [ERROR] Task 3 training failed. Stop.
    pause
    exit /b 1
)

REM ---------- Copy Task 3 required snapshots ----------
if exist ".\results_task3\task3_600000.pt" (
    copy /Y ".\results_task3\task3_600000.pt" "..\LAB5_H34116122_task3_600000.pt"
) else (
    echo [WARN] Missing .\results_task3\task3_600000.pt
)

if exist ".\results_task3\task3_1000000.pt" (
    copy /Y ".\results_task3\task3_1000000.pt" "..\LAB5_H34116122_task3_1000000.pt"
) else (
    echo [WARN] Missing .\results_task3\task3_1000000.pt
)

if exist ".\results_task3\task3_1500000.pt" (
    copy /Y ".\results_task3\task3_1500000.pt" "..\LAB5_H34116122_task3_1500000.pt"
) else (
    echo [WARN] Missing .\results_task3\task3_1500000.pt
)

if exist ".\results_task3\task3_2000000.pt" (
    copy /Y ".\results_task3\task3_2000000.pt" "..\LAB5_H34116122_task3_2000000.pt"
) else (
    echo [WARN] Missing .\results_task3\task3_2000000.pt
)

if exist ".\results_task3\task3_2500000.pt" (
    copy /Y ".\results_task3\task3_2500000.pt" "..\LAB5_H34116122_task3_2500000.pt"
) else (
    echo [WARN] Missing .\results_task3\task3_2500000.pt
)

if exist ".\results_task3\task3_best.pt" (
    copy /Y ".\results_task3\task3_best.pt" "..\LAB5_H34116122_task3_best.pt"
    echo [OK] Copied Task 3 best model to ..\LAB5_H34116122_task3_best.pt
) else if exist ".\results_task3\best_model.pt" (
    copy /Y ".\results_task3\best_model.pt" "..\LAB5_H34116122_task3_best.pt"
    echo [OK] Copied Task 3 best_model.pt to ..\LAB5_H34116122_task3_best.pt
) else (
    echo [WARN] Task 3 best model not found.
)

echo.
echo ============================================================
echo All training commands finished.
echo Please verify models with test_model1.py and test_pong.py.
echo ============================================================
pause
endlocal
