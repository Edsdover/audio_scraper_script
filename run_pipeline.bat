@echo off
setlocal enabledelayedexpansion

REM ==============================
REM   Podcast Pipeline (Windows)
REM ==============================

REM Change to script directory
cd /d "%~dp0"

REM Create venv if missing
if not exist "venv" (
  echo [INFO] Virtual environment not found. Creating one...
  py -3.11 -m venv venv
)

REM Upgrade pip only if quite old; otherwise leave as-is
call ".\venv\Scripts\python.exe" -m pip --version >NUL 2>&1
if errorlevel 1 (
  echo [INFO] Bootstrapping pip...
  call ".\venv\Scripts\python.exe" -m ensurepip
)

echo [INFO] Installing pinned dependencies...
call ".\venv\Scripts\python.exe" -m pip install --upgrade pip
call ".\venv\Scripts\python.exe" -m pip install -r requirements_windows.txt

echo.
echo Running pipeline...
echo.

call ".\venv\Scripts\python.exe" podcast_pipeline_gpu.py

echo.
echo === Pipeline finished ===
pause
endlocal
