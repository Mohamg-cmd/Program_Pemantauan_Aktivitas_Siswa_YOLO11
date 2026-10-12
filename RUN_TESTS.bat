@echo off
setlocal
cd /d "%~dp0"
python -m pytest -q tests
set EXITCODE=%ERRORLEVEL%
echo.
echo Pengujian selesai dengan kode: %EXITCODE%
pause
exit /b %EXITCODE%
