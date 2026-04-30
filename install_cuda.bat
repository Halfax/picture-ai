@echo off
setlocal enableextensions

REM ------------------------------------------------------------
REM PictureAI CUDA setup (NVIDIA boxes)
REM - Creates Python 3.11 venv in this folder
REM - Installs CUDA PyTorch stack from official index
REM - Installs PictureAI requirements
REM ------------------------------------------------------------

REM Create venv with Python 3.11 if it does not exist
if not exist "venv" (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv venv
)

call "%~dp0venv\Scripts\activate.bat"

python -m ensurepip --upgrade
python -m pip install --upgrade pip

REM Install CUDA-enabled PyTorch stack (let PyTorch pick best CUDA build)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Install remaining dependencies for Picture AI
python -m pip install -r requirements.txt

echo.
echo CUDA environment setup complete. To use it later, run:
echo   venv\Scripts\activate
endlocal
