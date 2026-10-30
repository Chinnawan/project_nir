@echo off
rem Local demo (Gradio). Opens http://127.0.0.1:7860 -- only on this machine (CAPA images are licensed by ULG; never share=True).
rem Tabs: mode 1 (CAPA, out-of-fold) / mode 2 (upload) / belt / "step by step: Kleynen vs CNN" (G2) / "batch" (G3).
rem Needs the CNN checkpoints (outputs\dl\confirm_ckpt\, from run_confirm.bat + run_n5.bat) and the Kleynen GUI models
rem (models\kleynen_*.pkl) -- the latter are built here automatically the first time (~1.5 min, CPU).
rem Set CUDA_VISIBLE_DEVICES=-1 to force CPU.
cd /d "%~dp0"
.venv\Scripts\python.exe -m src.consent
if errorlevel 1 exit /b 1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
if not exist "models\kleynen_750_800_all.pkl" (
    echo Kleynen GUI models not found - building them once ...
    .venv\Scripts\python.exe -u -m src.kleynen_models
    if errorlevel 1 goto :fail
)
.venv\Scripts\python.exe -u -m src.demo_app
pause
exit /b 0
:fail
echo Building the Kleynen GUI models failed.
pause
exit /b 1
