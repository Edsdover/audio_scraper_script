@echo off
setlocal enabledelayedexpansion

REM ==============================
REM   Podcast Pipeline (Windows)
REM ==============================

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
copy /Y "podcast_pipeline_gpu.py" "%SYS_FOLDER%\" > NUL
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

echo.
echo Running pipeline...
echo.

call ".\%SYS_FOLDER%\venv\Scripts\python.exe" "%SYS_FOLDER%\podcast_pipeline_gpu.py"

echo.
echo === Pipeline finished ===
pause
endlocal
