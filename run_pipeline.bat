@echo off
setlocal enabledelayedexpansion

REM ==============================
REM   Audio Pipeline (Windows)
REM ==============================

REM Set console to UTF-8 to support emoji characters in output
chcp 65001 > NUL

REM Change to script directory
cd /d "%~dp0"

REM --- Define the new system subfolder ---
set SYS_FOLDER=sys

REM Create venv if missing inside the sys folder
if not exist "%SYS_FOLDER%\venv" (
  echo [INFO] Virtual environment not found. Creating one in '%SYS_FOLDER%\venv'...
  py -3.11 -m venv "%SYS_FOLDER%\venv"
)

REM --- Copy core files into the system subfolder ---
echo [INFO] Preparing system folder...
copy /Y "audio_pipeline_gpu.py" "%SYS_FOLDER%\" > NUL
copy /Y "requirements_windows.txt" "%SYS_FOLDER%\" > NUL

REM Check pip version
call ".\%SYS_FOLDER%\venv\Scripts\python.exe" -m pip --version >NUL 2>&1
if errorlevel 1 (
  echo [INFO] Bootstrapping pip...
  call ".\%SYS_FOLDER%\venv\Scripts\python.exe" -m ensurepip
)

echo [INFO] Installing pinned dependencies from %SYS_FOLDER%\requirements_windows.txt...
call ".\%SYS_FOLDER%\venv\Scripts\python.exe" -m pip install --upgrade pip
call ".\%SYS_FOLDER%\venv\Scripts\python.exe" -m pip install -r "%SYS_FOLDER%\requirements_windows.txt"

REM Create audio/cache/results directories if they don't exist
if not exist "audio_files" mkdir "audio_files"
if not exist "cache" mkdir "cache"
if not exist "results" mkdir "results"
echo [INFO] Created data folders (audio_files, cache, results).

REM Check for Hugging Face Token
if not defined HF_TOKEN (
  echo.
  echo [SETUP] Hugging Face token not found.
  echo To download the required speaker identification models, you need a free HF token.
  echo 1. Go to https://huggingface.co/settings/tokens
  echo 2. Create a token with the 'read' role.
  echo 3. Paste it here:
  set /p HF_TOKEN="Enter your Hugging Face Token: "
)

echo.
echo Running pipeline...
echo.
for /f "tokens=*" %%i in ('dir /b /s "%cd%\cache\pipeline_*.log"') do (
    echo "Log file created at: %%i"
)
call ".\%SYS_FOLDER%\venv\Scripts\python.exe" "%SYS_FOLDER%\audio_pipeline_gpu.py"

echo.
echo === Pipeline finished ===
pause
endlocal
