@echo off
rem PrivLab: double-click to start it and open it in your browser. Close this window to stop PrivLab.
rem With an AI assistant (MCP) you do not need this: PrivLab starts its page itself and gives you the link.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto notinstalled
echo Starting PrivLab... it opens in your browser in a few seconds. Keep this window open while you use it.
".venv\Scripts\python.exe" -m viewer.server --open
echo.
echo PrivLab has stopped. You can close this window.
pause
exit /b 0
:notinstalled
echo PrivLab is not installed yet. See "Getting started" in README.md.
pause
exit /b 1
