@echo off
title PoultryVision AI - FastAPI Backend (Port 8000)

set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "%~dp0backend\.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%~dp0backend\.venv\Scripts\python.exe"
    ) else (
        echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe or backend\.venv\Scripts\python.exe
        echo Please create a virtual environment at project root: python -m venv .venv
        pause
        exit /b 1
    )
)

cd /d "%~dp0backend"
echo Starting FastAPI backend on http://127.0.0.1:8000 ...
"%PYTHON_EXE%" run.py

