@echo off
echo ============================================================
echo  HiSMD-Net Web Application Launcher
echo  AP50: 68.43%  mAP: 47.80%  APS: 30.23%
echo ============================================================
echo.

:: Install dependencies if needed
echo [1/2] Checking dependencies...
pip install fastapi uvicorn python-multipart torch torchvision opencv-python numpy pillow -q

echo [2/2] Starting HiSMD-Net server at http://localhost:8000
echo       Press Ctrl+C to stop.
echo.

python app.py

pause
