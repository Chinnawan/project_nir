@echo off
rem Overnight multi-seed run (resumable). Safe to close and re-run: finished (variant,seed,fold) are skipped.
rem   run_overnight.bat            full batch, then prints the summary report
rem   run_overnight.bat --smoke    tiny test (2 variants, 1 fold, 2 epochs) into a separate json
cd /d "%~dp0"
.venv\Scripts\python.exe -m src.consent
if errorlevel 1 exit /b 1
set PYTHONHASHSEED=0
set PYTHONUNBUFFERED=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8

echo ================================================================
echo  Overnight run started %date% %time%
echo  Log:     outputs\logs\overnight_YYYYMMDD.log
echo  Results: outputs\dl\multiseed_results.json  (written after every fold)
echo  Plug in the laptop. Ctrl+C stops safely; re-run this file to resume.
echo ================================================================

.venv\Scripts\python.exe -u -m src.run_multiseed %*
set RC=%ERRORLEVEL%
echo.
echo run_multiseed finished with exit code %RC% at %date% %time%

if "%~1"=="" (
  echo Writing summary report to outputs\logs\overnight_report.txt ...
  .venv\Scripts\python.exe -u -m src.multiseed_report > outputs\logs\overnight_report.txt 2>&1
  type outputs\logs\overnight_report.txt
  echo.
  echo Done. Summary json: outputs\dl\multiseed_summary.json
  pause
)
exit /b %RC%
