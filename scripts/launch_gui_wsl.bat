@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%A in ("%SCRIPT_DIR%..") do set "REPO_DIR=%%~fA"

for /f "delims=" %%W in ('wsl.exe -e bash -lc "wslpath -u \"%REPO_DIR%\" "') do set "WSL_REPO=%%W"

if "%WSL_REPO%"=="" (
  echo Failed to resolve WSL path for "%REPO_DIR%".
  exit /b 1
)

wsl.exe -e bash -lc "cd \"%WSL_REPO%\" && bash scripts/launch_gui.sh"

endlocal
