@echo off
rem N5: visible_ss + full_ss (aligned SingleSpectra images), seeds 3,4,5 x 5 folds, checkpoints for the demo.
rem Resumable: finished (variant,seed,fold) are skipped.
rem   run_n5.bat            full run + checkpoint verification, then the N5 analysis
rem   run_n5.bat --smoke    tiny test into separate files
cd /d "D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase"
set PYTHONHASHSEED=0
set PYTHONUNBUFFERED=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8

echo ================================================================
echo  N5 run started %date% %time%
echo  Log:         outputs\logs\n5_YYYYMMDD.log
echo  Results:     outputs\dl\n5_results.json  (written after every fold)
echo  Checkpoints: outputs\dl\confirm_ckpt\{visible_ss,full_ss}_seed*_fold*.pt
echo  Ctrl+C stops safely; re-run this file to resume.
echo ================================================================

.venv\Scripts\python.exe -u -m src.run_n5 %*
set RC=%ERRORLEVEL%
echo run_n5 finished with exit code %RC% at %date% %time%
if "%~1"=="" .venv\Scripts\python.exe -u -m src.nir_proof n5
exit /b %RC%
