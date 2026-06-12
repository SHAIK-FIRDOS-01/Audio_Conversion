@echo off
echo ===================================================
echo   RVC Voice Conversion Pipeline Setup (Windows)
echo ===================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system!
    echo Please install Python 3.10 or newer and ensure it is added to your PATH.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating Python virtual environment venv...
    python -m venv venv || (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
) else (
    echo [INFO] Virtual environment 'venv' already exists.
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Downgrade pip to support legacy package metadata (required for omegaconf/rvc-python)
echo [INFO] Installing compatible pip version...
python -m pip install "pip<24.1"

:: Install PyTorch
echo [INFO] Installing PyTorch and Torchaudio...
:: Installing the CPU wheel is fast and stable for environments without an Nvidia GPU.
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Install core dependencies
echo [INFO] Installing RVC, Gradio, and Google APIs...
python -m pip install rvc-python gradio google-api-python-client google-auth-oauthlib google-auth-httplib2

:: Verify installation / install fallback for pyworld if needed
if %errorlevel% neq 0 (
    echo [WARNING] Main installation encountered warnings. Trying fallback for pyworld...
    python -m pip install pyworld-prebuilt
)

:: Create folder structure
echo [INFO] Creating folder structure...
if not exist "inputs" mkdir inputs
if not exist "outputs" mkdir outputs
if not exist "models" mkdir models

echo.
echo ===================================================
echo   Setup completed successfully!
echo   Place your audio files in 'inputs/' and run 'run.bat'.
echo ===================================================
pause
