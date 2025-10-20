@echo off
REM =============================================================
REM Paper configuration batch training script (Windows)
REM -------------------------------------------------------------
REM Description:
REM   - Train DDDQN-PER and Baseline-DQN with paper config
REM   - Episode length: 17,520 steps (6 months real data)
REM   - Training steps: 1,200,000 steps/seed
REM   - 4 random seeds: [0, 1, 2, 3]
REM   - Other parameters follow paper table config
REM =============================================================

setlocal enabledelayedexpansion

REM ----------------- Parameter config -----------------
set SEEDS=0 1 2 3
set STEPS=1200000
set EPISODE_LENGTH=17520

REM Data file paths
set DATA_LOAD=data_processed\load_15min_fullyear.parquet
set DATA_PRICE=data_processed\price_15min_fullyear.parquet
set EV_DEMAND=data_processed\ev_demand_15min_fullyear.parquet
set SOH_JSON=data_processed\soh_params_fullyear.json

REM Output directories
set OUTPUT_DIR=models
set RESULTS_DIR=results
REM -------------------------------------------

REM Create output directories
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

REM Configuration summary
echo =============================================================
echo Paper configuration training script
echo =============================================================
echo Seeds        : %SEEDS%
echo Steps        : %STEPS%
echo Episode Length: %EPISODE_LENGTH%
echo Output Dir   : %OUTPUT_DIR%
echo Results Dir  : %RESULTS_DIR%
echo -------------------------------------------------------------
echo Load Data    : %DATA_LOAD%
echo Price Data   : %DATA_PRICE%
echo EV Demand    : %EV_DEMAND%
echo SOH Params   : %SOH_JSON%
echo =============================================================

REM Data integrity check
if not exist "%DATA_LOAD%" (
    echo Missing data file: %DATA_LOAD%
    exit /b 1
)
if not exist "%DATA_PRICE%" (
    echo Missing data file: %DATA_PRICE%
    exit /b 1
)
if not exist "%EV_DEMAND%" (
    echo Missing data file: %EV_DEMAND%
    exit /b 1
)
if not exist "%SOH_JSON%" (
    echo Missing data file: %SOH_JSON%
    exit /b 1
)

echo Data files check passed

REM Train DDDQN-PER
echo.
echo Training DDDQN-PER (paper config)
echo =============================================================

for %%s in (%SEEDS%) do (
    echo.
    echo ^>^>^> Training DDDQN-PER SEED=%%s
    echo -------------------------------------------------------------
    
    python code\train_dddqn_paper.py ^
        --load "%DATA_LOAD%" ^
        --price "%DATA_PRICE%" ^
        --ev "%EV_DEMAND%" ^
        --soh "%SOH_JSON%" ^
        --save "%OUTPUT_DIR%\dddqn_per_paper_seed_%%s.zip" ^
        --seed %%s ^
        --steps %STEPS% ^
        --episode_length %EPISODE_LENGTH%
    
    if errorlevel 1 (
        echo DDDQN-PER SEED=%%s training failed
        exit /b 1
    )
    
    echo ^<^<^< DDDQN-PER SEED=%%s training completed
    echo ----------------------------------------
    timeout /t 5 /nobreak >nul
)

REM Train Baseline-DQN
echo.
echo Training Baseline-DQN (paper config)
echo =============================================================

for %%s in (%SEEDS%) do (
    echo.
    echo ^>^>^> Training Baseline-DQN SEED=%%s
    echo -------------------------------------------------------------
    
    python code\baseline_dqn_train.py ^
        --load "%DATA_LOAD%" ^
        --price "%DATA_PRICE%" ^
        --ev "%EV_DEMAND%" ^
        --soh "%SOH_JSON%" ^
        --save "%OUTPUT_DIR%\baseline_dqn_paper_seed_%%s.zip" ^
        --seed %%s ^
        --steps %STEPS% ^
        --episode_length %EPISODE_LENGTH% ^
        --buffer 150000 ^
        --net_width 128
    
    if errorlevel 1 (
        echo Baseline-DQN SEED=%%s training failed
        exit /b 1
    )
    
    echo ^<^<^< Baseline-DQN SEED=%%s training completed
    echo ----------------------------------------
    timeout /t 5 /nobreak >nul
)

echo.
echo All training tasks completed!
echo =============================================================
echo Model files saved in: %OUTPUT_DIR%\
echo Training results saved in: %RESULTS_DIR%\
echo.
echo Training config summary:
echo   - DDDQN-PER: network width 256, buffer 200,000, batch size 32
echo   - Baseline-DQN: network width 128, buffer 150,000, batch size 32
echo   - Common config: learning rate 0.0003, gamma 0.99, target update 1000
echo =============================================================

pause 