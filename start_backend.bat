@echo off
echo ========================================
echo Starting Melanoma Detection Backend
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created.
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo.
echo Installing/updating dependencies...
pip install -r requirements.txt --quiet

REM Start backend
echo.
echo ========================================
echo Backend starting on http://localhost:8000
echo API docs: http://localhost:8000/docs
echo ========================================
echo.

cd backend
python app.py

pause



