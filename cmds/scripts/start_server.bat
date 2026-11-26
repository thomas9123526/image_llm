@echo off
REM Start Z-Image-Turbo API on Windows (RTX 5090). Edit the values below.
REM Run from anywhere:  cmds\scripts\start_server.bat
setlocal
set ROOT=%~dp0..\..
if exist "%ROOT%\.venv\Scripts\activate.bat" call "%ROOT%\.venv\Scripts\activate.bat"

set ZIMAGE_MODEL_PATH=%ROOT%\models\Z-Image-Turbo
set ZIMAGE_API_KEYS=change-me-key1,change-me-key2
set ZIMAGE_HOST=0.0.0.0
set ZIMAGE_PORT=8000
set ZIMAGE_QUEUE_MAX=32
set ZIMAGE_MAX_PER_KEY=4
set ZIMAGE_JOB_TTL=3600
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

cd /d "%ROOT%\zimage_service"
python server.py
endlocal
