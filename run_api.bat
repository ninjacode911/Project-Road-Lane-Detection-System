@echo off
REM Quick script to run the API server without installing the package

echo Starting Lane Detection API Server...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set PYTHONPATH to include current directory
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Run the API
python web\api.py

pause
