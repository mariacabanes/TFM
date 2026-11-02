@echo off
:: =============================================================================
::  setup_env.bat  —  One-shot environment setup for UNet / FL training
::  Usage:  Double-click or run from Command Prompt:  setup_env.bat
::
::  Requirements before running:
::    - Windows 10 (1709+) or Windows 11  (winget is built-in)
::    - NVIDIA GPU driver  (https://www.nvidia.com/drivers)
::      CUDA toolkit is bundled with PyTorch — no separate install needed
:: =============================================================================

setlocal enabledelayedexpansion

set VENV_DIR=envs\fl_env
set PYTHON=%VENV_DIR%\Scripts\python.exe
set PIP=%VENV_DIR%\Scripts\pip.exe

echo.
echo ==================================================
echo   UNet / FL Training  —  Environment Setup
echo ==================================================
echo.

:: =============================================================================
:: 1/6  Check and auto-install Python
:: =============================================================================
echo [1/6]  Checking Python...

python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
    echo [OK]  Python !PYVER! found.
    goto :python_ready
)

echo [!]   Python not found — attempting auto-install via winget...

:: Check winget is available
winget --version >nul 2>&1
if errorlevel 1 (
    echo [X]  winget not found. Your Windows version may be too old.
    echo      Install Python 3.10+ manually from https://www.python.org/downloads/
    echo      Tick "Add Python to PATH" during install, then re-run this script.
    pause
    exit /b 1
)

echo [i]   Installing Python 3.11 via winget (this may take a few minutes)...
winget install --id Python.Python.3.11 --source winget --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [X]  winget install failed.
    echo      Install Python 3.10+ manually from https://www.python.org/downloads/
    echo      Tick "Add Python to PATH" during install, then re-run this script.
    pause
    exit /b 1
)

echo [OK]  Python installed via winget.
echo [i]   Refreshing PATH so Python is visible in this session...

:: Refresh PATH from registry so we don't need a new terminal
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USERPATH=%%B"
for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PATH 2^>nul') do set "SYSPATH=%%B"
set "PATH=%SYSPATH%;%USERPATH%"

:: Verify Python is now callable
python --version >nul 2>&1
if errorlevel 1 (
    echo [!]   Python still not on PATH after install.
    echo       Please close this window, open a NEW Command Prompt, and re-run setup_env.bat.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK]  Python !PYVER! is ready.

:python_ready

:: =============================================================================
:: 2/6  Fix PowerShell execution policy
:: =============================================================================
echo.
echo [2/6]  Setting PowerShell execution policy...

powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force" >nul 2>&1
if errorlevel 1 (
    echo [!]   Could not set execution policy -- skipping.
) else (
    echo [OK]  PowerShell execution policy set to RemoteSigned.
)

:: =============================================================================
:: 3/6  Create virtual environment
:: =============================================================================
echo.
echo [3/7]  Creating virtual environment...

if exist "%VENV_DIR%" (
    echo [!]   Virtual environment already exists at %VENV_DIR% — skipping.
) else (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [X]  Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]  Virtual environment created at %VENV_DIR%
)

:: =============================================================================
:: 3/6  Upgrade pip
:: =============================================================================
echo.
echo [4/7]  Upgrading pip...

"%PYTHON%" -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [!]   pip upgrade had issues — continuing anyway.
) else (
    echo [OK]  pip upgraded.
)

:: =============================================================================
:: 4/6  Install PyTorch with CUDA 12.1
:: =============================================================================
echo.
echo [5/7]  Installing PyTorch with CUDA 12.1...
echo        This may take a few minutes (~3-5 GB download).

"%PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
if errorlevel 1 (
    echo [X]  PyTorch installation failed.
    pause
    exit /b 1
)
echo [OK]  PyTorch installed.

:: =============================================================================
:: 5/6  Install training dependencies + verify GPU
:: =============================================================================
echo.
echo [6/7]  Installing training dependencies...

"%PIP%" install opencv-contrib-python pycocotools scipy matplotlib tqdm numpy dill blosc torch_geometric pandas seaborn scikit-learn IPython --quiet

if errorlevel 1 (
    echo [X]  Dependency installation failed.
    pause
    exit /b 1
)
echo [OK]  Dependencies installed.

echo.
echo        Checking CUDA availability...
"%PYTHON%" -c "import torch; cuda=torch.cuda.is_available(); print('[OK]  CUDA available :',cuda); print('[OK]  GPU            :',torch.cuda.get_device_name(0) if cuda else 'none'); print('[OK]  CUDA version   :',torch.version.cuda if cuda else 'n/a')"

echo.
echo        Saving requirements.txt...
"%PIP%" freeze > requirements.txt
echo [OK]  requirements.txt saved.

:: =============================================================================
:: 6/6  Create input_zips directory
:: =============================================================================
echo.
echo [7/7]  Creating input_zips directory...
if exist "input_zips" (
    echo [!]   input_zips directory already exists — skipping.
) else (
    mkdir input_zips
    if errorlevel 1 (
        echo [X]  Failed to create input_zips directory.
        pause
        exit /b 1
    )
    echo [OK]  input_zips directory created.
)

:: =============================================================================
echo.
echo ==================================================
echo   Setup complete!
echo ==================================================
echo.
echo   To activate the environment in a new terminal:
echo.
echo     %VENV_DIR%\Scripts\activate
echo.
echo   Then run the pipeline:
echo.
echo     python "(0) run_all.py"
echo.
pause
