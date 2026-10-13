@echo off
rem Round 2 overnight batch (resumable). Safe to close and re-run: finished (variant,seed,fold) are skipped.
rem   run_round2.bat            full batch, then Grad-CAM 14x14, then the summary report
rem   run_round2.bat --smoke    tiny test (every variant, 1 fold, 2 epochs) into a separate json
cd /d "D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase"
set PYTHONHASHSEED=0
set PYTHONUNBUFFERED=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8

echo ================================================================
echo  Round 2 started %date% %time%
echo  Log:     outputs\logs\round2_YYYYMMDD.log
echo  Results: outputs\dl\round2_results.json  (written after every fold)
echo  Plug in the laptop. Ctrl+C stops safely; re-run this file to resume.
echo ================================================================

.venv\Scripts\python.exe -u -m src.run_round2 %*
set RC=%ERRORLEVEL%
echo.
echo run_round2 finished with exit code %RC% at %date% %time%

if "%~1"=="" (
  if not exist outputs\dl\gradcam_mask_eval_14x14.json (
    echo Step 5: Grad-CAM 14x14 on the nir_SE_ss checkpoints ...
    .venv\Scripts\python.exe -u -m src.gradcam_mask_eval --layer 5 --out outputs\dl\gradcam_mask_eval_14x14.json --overlay-dir outputs\dl\gradcam_mask_14x14 > outputs\logs\gradcam_14x14.txt 2>&1
  )
  echo Writing summary report to outputs\logs\round2_report.txt ...
  .venv\Scripts\python.exe -u -m src.round2_report > outputs\logs\round2_report.txt 2>&1
  type outputs\logs\round2_report.txt
  echo.
  echo Done. Summary json: outputs\dl\round2_summary.json
  pause
)
exit /b %RC%
