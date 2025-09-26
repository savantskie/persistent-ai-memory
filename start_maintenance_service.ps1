# Friday Automatic Maintenance Service
# PowerShell script to run the automatic database maintenance service

Write-Host "Starting Friday Automatic Database Maintenance..." -ForegroundColor Green
Write-Host ""
Write-Host "This service will:" -ForegroundColor Cyan
Write-Host "- Run database cleanup every 3 hours" -ForegroundColor White
Write-Host "- Optimize database performance" -ForegroundColor White  
Write-Host "- Remove old data according to retention policies" -ForegroundColor White
Write-Host "- Prevent database bloat" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Yellow
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

try {
    # Start the maintenance service
    python friday_auto_maintenance.py --interval 3
}
catch {
    Write-Host "Error starting maintenance service: $_" -ForegroundColor Red
}
finally {
    Write-Host ""
    Write-Host "Friday Maintenance Service stopped." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
