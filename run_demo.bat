@echo off
REM Quick demo script using the simple quick_start.py

echo Lane Detection Quick Demo
echo.

if "%~1"=="" (
    echo Usage: run_demo.bat ^<image_path^>
    echo.
    echo Example:
    echo   run_demo.bat data\test_image.jpg
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Run quick start demo
python quick_start.py %1 %2

pause
