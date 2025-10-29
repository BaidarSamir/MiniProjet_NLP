@echo off
REM Disease Predictor - Setup Script
REM Installs all required dependencies

echo ================================================
echo Disease Prediction System - Setup
echo ================================================
echo.
echo This script will install all required Python packages.
echo Make sure you have Python 3.8+ installed.
echo.
pause

echo.
echo [1/5] Installing core dependencies...
pip install pandas numpy scikit-learn matplotlib seaborn

echo.
echo [2/5] Installing deep learning libraries...
pip install tensorflow keras

echo.
echo [3/5] Installing audio processing libraries...
pip install sounddevice scipy

echo.
echo [4/5] Installing NLP libraries...
pip install sentence-transformers

echo.
echo [5/5] Installing API client libraries...
pip install elevenlabs python-dotenv openai

echo.
echo ================================================
echo Setup complete!
echo ================================================
echo.
echo Next steps:
echo 1. Configure your API keys in .env file
echo 2. Open Disease_predictor_net.ipynb to train the model
echo 3. Or open speech_to_text.ipynb to test voice input
echo.
pause
