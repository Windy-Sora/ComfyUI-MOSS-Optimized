@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-CUDA13.ps1" %*
pause
