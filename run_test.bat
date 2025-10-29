@echo off
REM Disease Predictor - Quick Test Runner
REM Make sure you've trained the model first!

echo ================================================
echo Disease Prediction System - Quick Test
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [INFO] Python found
echo.

REM Check if required files exist
if not exist "symbipredict_2022.csv" (
    echo [ERROR] symbipredict_2022.csv not found!
    echo Make sure you're in the project directory.
    pause
    exit /b 1
)

echo [INFO] Dataset found
echo.

REM Check if model exists
if not exist "disease_model.h5" (
    echo [WARNING] No trained model found!
    echo.
    echo Please train the model first:
    echo 1. Open Disease_predictor_net.ipynb
    echo 2. Run all cells to train the model
    echo 3. Add a cell at the end with: symbi_model.save('disease_model.h5'^)
    echo 4. Run this script again
    echo.
    pause
    exit /b 1
)

echo [INFO] Model found
echo.
echo [INFO] Starting test...
echo ================================================
echo.

REM Run the test script
python test_disease_predictor.py

echo.
echo ================================================
echo Test completed!
echo ================================================
pause
