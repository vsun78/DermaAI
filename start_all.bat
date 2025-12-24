@echo off
echo ========================================
echo Starting Melanoma Detection System
echo ========================================
echo.
echo This will start both backend and frontend in separate windows.
echo.

REM Start backend in new window
start "Melanoma Detection - Backend" cmd /k start_backend.bat

REM Wait a few seconds for backend to initialize
echo Waiting for backend to initialize...
timeout /t 5 /nobreak

REM Start frontend in new window
start "Melanoma Detection - Frontend" cmd /k start_frontend.bat

echo.
echo ========================================
echo Both services are starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo ========================================
echo.
echo You can close this window.



