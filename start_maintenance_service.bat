@echo off
REM Friday Automatic Maintenance Service
REM Run this to start the automatic database maintenance service

echo Starting Friday Automatic Database Maintenance...
echo.
echo This service will:
echo - Run database cleanup every 3 hours
echo - Optimize database performance
echo - Remove old data according to retention policies
echo - Prevent database bloat
echo.
echo Press Ctrl+C to stop the service
echo.

cd /d "%~dp0"
python friday_auto_maintenance.py --interval 3

echo.
echo Friday Maintenance Service stopped.
pause
