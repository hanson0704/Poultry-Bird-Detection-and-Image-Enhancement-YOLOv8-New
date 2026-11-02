@echo off
title PoultryVision AI - React Frontend (Port 5173)
cd /d "%~dp0frontend"
if not exist "node_modules" (
    echo [ERROR] node_modules not found in frontend\. Run "npm install" inside frontend\ first.
    pause
    exit /b 1
)
echo Starting Vite development server on http://127.0.0.1:5173 ...
call npm run dev

