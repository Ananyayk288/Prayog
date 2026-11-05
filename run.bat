@echo off
REM ============================================================================
REM EV Motor Fault Detection System — One-Click Dashboard Launcher (Windows)
REM ============================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment .venv not found!
    echo Please create it first: py -m venv .venv ^&^& .\.venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

if not exist "models\saved\random_forest_model.joblib" (
    echo [INFO] Training initial ML models on synthetic demonstration dataset...
    python scripts\train.py
)

echo [INFO] Starting EV Motor Health Monitoring Streamlit Dashboard...
streamlit run dashboard/app.py

