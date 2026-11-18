@echo off
title FX Reborn Toolkit Ultra v3.0 - Interface Grafica
cd /d "%~dp0"
if exist "python\python.exe" (
    "python\python.exe" fxkit.py gui
    goto end
)
py -3.13 -c "" >nul 2>&1
if not errorlevel 1 (
    py -3.13 fxkit.py gui
    goto end
)
python fxkit.py gui
:end
if errorlevel 1 pause
