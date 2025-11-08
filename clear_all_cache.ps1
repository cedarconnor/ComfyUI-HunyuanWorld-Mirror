# Clear all Python cache aggressively
Write-Host "Clearing all Python cache files..." -ForegroundColor Yellow

# Clear __pycache__ directories
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Clear all .pyc files
Get-ChildItem -Path . -Recurse -Include *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue

# Clear all .pyo files
Get-ChildItem -Path . -Recurse -Include *.pyo | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host "Cache cleared!" -ForegroundColor Green
Write-Host ""
Write-Host "Now:" -ForegroundColor Cyan
Write-Host "1. Close ComfyUI completely (make sure process is stopped)" -ForegroundColor White
Write-Host "2. Restart ComfyUI" -ForegroundColor White
Write-Host "3. Enable 'force_reload' checkbox in LoadHunyuanWorldMirrorModel node" -ForegroundColor White
Write-Host "4. Run your workflow" -ForegroundColor White
