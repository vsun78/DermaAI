@echo off
echo ========================================
echo Starting Melanoma Detection Frontend
echo ========================================
echo.

cd frontend

REM Check if node_modules exists
if not exist "node_modules\" (
    echo Installing dependencies...
    npm install
    echo.
)

echo ========================================
echo Frontend starting on http://localhost:3000
echo ========================================
echo.

npm start

pause



