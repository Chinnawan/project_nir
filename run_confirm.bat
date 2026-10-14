@echo off
rem Confirmatory run: seeds 3,4,5 for nir_ss (locked formula) and nir (baseline) + exploratory nir_ss_ls.
rem Resumable: finished (variant,seed,fold) are skipped.
rem   run_confirm.bat            full run, then the report
rem   run_confirm.bat --smoke    tiny test into separate files
cd /d "D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase"
set PYTHONHASHSEED=0
set PYTHONUNBUFFERED=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8

echo ================================================================
echo  Confirmatory run started %date% %time%
echo  Log:         outputs\logs\confirm_YYYYMMDD.log
echo  Results:     outputs\dl\confirm_results.json  (written after every fold)
echo  Checkpoints: outputs\dl\confirm_ckpt\  (nir_ss, 15 files)
echo  Plug in the laptop. Ctrl+C stops safely; re-run this file to resume.
echo ================================================================

.venv\Scripts\python.exe -u -m src.run_confirm %*
set RC=%ERRORLEVEL%
echo.
echo run_confirm finished with exit code %RC% at %date% %time%

if "%~1"=="" (
  echo Writing report to outputs\logs\confirm_report.txt ...
  .venv\Scripts\python.exe -u -m src.confirm_report > outputs\logs\confirm_report.txt 2>&1
  type outputs\logs\confirm_report.txt
  echo.
  echo Done. Summary json: outputs\dl\confirm_summary.json
  pause
)
exit /b %RC%
