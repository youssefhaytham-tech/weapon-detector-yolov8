@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please set up Python first.
    pause
    exit /b 1
)
call "%~dp0venv\Scripts\activate.bat"
"%~dp0venv\Scripts\python.exe" "%~dp0detect_gui.py"
if errorlevel 1 (
    pause
)
