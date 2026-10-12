@echo off
setlocal
cd /d "%~dp0"
python run_pipeline.py --config config.yaml %*
set EXITCODE=%ERRORLEVEL%
echo.
echo Program selesai dengan kode: %EXITCODE%
pause
exit /b %EXITCODE%
