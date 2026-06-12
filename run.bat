@echo off
echo ===================================================
echo   Starting local RVC Voice Conversion Web UI...
echo ===================================================
echo.

if not exist "venv" (
    echo [ERROR] Virtual environment 'venv' was not found!
    echo Please run 'setup.bat' first to install dependencies.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Start Web UI
echo [INFO] Starting app.py...
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed or exited with error code %errorlevel%
    pause
)
