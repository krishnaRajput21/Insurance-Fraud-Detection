@echo off
setlocal
echo ===================================================
echo Automobile Insurance Fraud Detection System
echo ===================================================

echo.
echo Checking for Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Downloading Python 3.11...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe' -OutFile 'python_installer.exe'"
    if exist python_installer.exe (
        echo Installing Python. Please wait... This might take a few minutes.
        start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
        echo Python installation finished!
        del python_installer.exe
        
        echo.
        echo ACTION REQUIRED: Python has been installed.
        echo Please CLOSE THIS WINDOW AND RE-RUN setup.bat to apply system path changes.
        pause
        exit /b
    ) else (
        echo Failed to download Python. Please install it manually from python.org
        pause
        exit /b
    )
) else (
    echo Python is already installed.
)

echo.
echo Step 1: Installing Python Dependencies...
pip install -r requirements.txt

echo.
echo Step 2: Generating Dataset and Training Models...
python model_training.py

echo.
echo Step 3: Starting the Web Application...
echo Please look for the URL (e.g., http://127.0.0.1:5000) below and open it!
echo.
python app.py

pause
