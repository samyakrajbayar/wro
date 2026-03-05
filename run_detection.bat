@echo off
setlocal EnableDelayedExpansion
echo ============================================================
echo   LEGO Detection - Unified Launcher (With Auto-Setup)
echo ============================================================
echo.

:: Configuration
set PRETRAINED_ONLINE=pretrained\detect_online.py
set PRETRAINED_LOCAL=pretrained\detect_yolo_generic.py
set CUSTOM_MODEL=runs/detect/lego_model_v1/weights/best.pt
set DETECT_SCRIPT=detect.py
set API_KEY=AIzaSyAGFK-4068Wy_iwqyrNm6v6VceQAY2M4fk
set GOOGLE_API_KEY=%API_KEY%

:: Auto-Install Libraries
echo [1/2] Checking for required libraries...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Error: pip not found. Please ensure Python is installed and added to PATH.
    pause
    exit /b
)

:: Run pip install (fast if already installed)
echo [!] Ensuring dependencies are up to date...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [!] Warning: Some dependencies failed to install automatically. 
    echo     Ensure you have an internet connection and trial again.
)

:: Check if custom model exists
if exist "%CUSTOM_MODEL%" (
    set CUSTOM_STATUS=[READY]
) else (
    set CUSTOM_STATUS=[NOT TRAINED YET]
)

echo.
echo [2/2] Please select a detection mode:
echo.
echo [1] Online (Google Gemini 2.0 Flash) - Most Accurate
echo [2] Offline Generic (YOLOv8m) - Baseline accuracy
echo [3] Custom Offline Model %CUSTOM_STATUS% - Trained on WRO rules
echo [4] Exit
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Launching Online Mode...
    if exist venv\Scripts\activate (call venv\Scripts\activate)
    python %PRETRAINED_ONLINE% --source 0
)

if "%choice%"=="2" (
    echo.
    echo Launching Offline Generic Mode...
    if exist venv\Scripts\activate (call venv\Scripts\activate)
    python %PRETRAINED_LOCAL% --source 0
)

if "%choice%"=="3" (
    if exist "%CUSTOM_MODEL%" (
        echo.
        echo Launching Custom Offline Model...
        if exist venv\Scripts\activate (call venv\Scripts\activate)
        python %DETECT_SCRIPT% --weights %CUSTOM_MODEL%
    ) else (
        echo.
        echo [X] Error: The custom model is not trained yet.
        echo Please run setup_and_train.bat first or wait for the background training to finish.
    )
)

if "%choice%"=="4" exit /b

echo.
echo Press any key to return to menu...
pause >nul
cls
%0
