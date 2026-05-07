@echo off
chcp 65001 >nul

echo ========================================
echo DLP Lab5 Evaluation for Submission
echo Student ID: H34116122
echo ========================================

REM 啟用虛擬環境
call .venv\Scripts\activate

echo.
echo ========================================
echo [Step 0] Check model files in parent folder
echo ========================================
dir /b ..\LAB5_H34116122*.pt

echo.
echo ========================================
echo [Step 1] Copy best Task 3 snapshot
echo ========================================

copy /Y ..\LAB5_H34116122_task3_2500000.pt ..\LAB5_H34116122_task3_best.pt

echo.
echo Task 3 required files should include:
echo ..\LAB5_H34116122_task3_600000.pt
echo ..\LAB5_H34116122_task3_1000000.pt
echo ..\LAB5_H34116122_task3_1500000.pt
echo ..\LAB5_H34116122_task3_2000000.pt
echo ..\LAB5_H34116122_task3_2500000.pt
echo ..\LAB5_H34116122_task3_best.pt

echo.
echo ========================================
echo [Task 1] Evaluate CartPole model, seeds 0 to 19
echo ========================================
python test_model1.py --model-path ..\LAB5_H34116122_task1.pt

echo.
echo ========================================
echo [Task 2] Evaluate Pong Vanilla DQN model, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task2.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo [Task 3 Snapshot] Evaluate 600000 steps, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task3_600000.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo [Task 3 Snapshot] Evaluate 1000000 steps, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task3_1000000.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo [Task 3 Snapshot] Evaluate 1500000 steps, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task3_1500000.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo [Task 3 Snapshot] Evaluate 2000000 steps, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task3_2000000.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo [Task 3 Snapshot] Evaluate 2500000 steps, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task3_2500000.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo [Task 3 Best] Evaluate copied best model, seeds 0 to 19
echo ========================================
python test_pong.py ^
  --model-path ..\LAB5_H34116122_task3_best.pt ^
  --episodes 20 ^
  --seed 0 ^
  --max-episode-steps 10000

echo.
echo ========================================
echo All evaluations finished.
echo Please screenshot the terminal results.
echo ========================================

pause