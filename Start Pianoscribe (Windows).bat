@echo off
rem Double-click this file to open Pianoscribe (Windows).
cd /d "%~dp0"
title Pianoscribe

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (python --version >nul 2>nul && set "PY=python")
if not defined PY goto nopython

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import pianoscribe.app" >nul 2>nul && goto run
)
echo First start: setting up Pianoscribe. This takes a few minutes and only happens once...
%PY% -m venv .venv || goto fail
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip || goto fail
".venv\Scripts\python.exe" -m pip install --quiet -e . || goto fail

:run
".venv\Scripts\python.exe" -m pianoscribe.app
goto :eof

:nopython
echo Python is not installed.
echo Download it from https://www.python.org/downloads/ - during setup tick "Add python.exe to PATH" -
echo then double-click this file again.
start "" "https://www.python.org/downloads/"
pause
goto :eof

:fail
echo.
echo Setup failed - see the messages above.
pause
