@echo off
REM =============================================================================
REM Lane Detection System - Easy Setup Script
REM =============================================================================

echo.
echo ========================================
echo   Lane Detection System - Setup
echo   2026 Edition
echo ========================================
echo.

REM Check Python version
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo [1/5] Creating virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/5] Installing dependencies...
echo This may take several minutes...
pip install -r requirements-2026.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Some dependencies may have failed to install
    echo Trying with CPU-only versions...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements-2026.txt
)

echo.
echo [5/5] Creating necessary directories...
if not exist "data\temp" mkdir data\temp
if not exist "data\uploads" mkdir data\uploads
if not exist "results\images" mkdir results\images
if not exist "results\videos" mkdir results\videos
if not exist "results\masks" mkdir results\masks
if not exist "results\logs" mkdir results\logs
if not exist "models\weights" mkdir models\weights

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the web interface:
echo   python web\api.py
echo   Then open: http://localhost:8000
echo.
echo To process an image:
echo   python app\main_modern.py path\to\image.jpg
echo.
echo ========================================

pause
