@echo off
rem Local demo (Gradio). Opens http://127.0.0.1:7860 -- only on this machine (CAPA images are licensed by ULG).
rem Needs the checkpoints from run_confirm.bat (outputs\dl\confirm_ckpt\). Set CUDA_VISIBLE_DEVICES=-1 to force CPU.
cd /d "D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase"
set PYTHONUNBUFFERED=1
.venv\Scripts\python.exe -u -m src.demo_app
pause
