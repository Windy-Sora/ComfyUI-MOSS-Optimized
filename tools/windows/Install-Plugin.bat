@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Plugin.ps1" %*
pause
