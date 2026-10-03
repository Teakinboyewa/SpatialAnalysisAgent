@echo off
rem Start QGIS with a debugpy port open, so VS Code can attach ("Attach to QGIS").
rem Needs debugpy in QGIS's Python and the startup.py block described in dev\README.md.
rem
rem QGIS is found automatically under "C:\Program Files\QGIS *"; set QGIS_BAT to
rem use another installation, e.g.  set QGIS_BAT=D:\QGIS\bin\qgis-ltr.bat

set QGIS_DEBUGPY=1
rem Only standard-library breakpoints are affected by frozen modules; hide the warning
set PYDEVD_DISABLE_FILE_VALIDATION=1
if "%QGIS_DEBUGPY_PORT%"=="" set QGIS_DEBUGPY_PORT=5678

if "%QGIS_BAT%"=="" (
    for /d %%D in ("C:\Program Files\QGIS *") do (
        if exist "%%~D\bin\qgis-ltr.bat" set "QGIS_BAT=%%~D\bin\qgis-ltr.bat"
        if exist "%%~D\bin\qgis.bat" if not exist "%%~D\bin\qgis-ltr.bat" set "QGIS_BAT=%%~D\bin\qgis.bat"
    )
)
if "%QGIS_BAT%"=="" (
    echo Could not find QGIS. Set QGIS_BAT to your qgis-ltr.bat or qgis.bat.
    exit /b 1
)

echo Starting QGIS for debugging: %QGIS_BAT%
start "" "%QGIS_BAT%"

rem Wait (up to 3 minutes) until QGIS has opened the debug port, so VS Code can attach right away.
powershell -NoProfile -Command "$deadline=(Get-Date).AddMinutes(3); while((Get-Date) -lt $deadline){ try { $c=New-Object Net.Sockets.TcpClient('127.0.0.1', %QGIS_DEBUGPY_PORT%); $c.Close(); Write-Host 'QGIS is ready for debugging.'; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; Write-Host 'QGIS did not open the debug port. See dev\README.md.'; exit 1"
