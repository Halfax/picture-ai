@echo off
setlocal enableextensions

REM ------------------------------------------------------------
REM PictureAI ROCm setup (self-contained)
REM - Creates %USERPROFILE%\rocm-pytorch if missing
REM - Downloads known ROCm wheels from TheRock GitHub release
REM - Creates Python 3.12 venv in this folder
REM - Installs ROCm torch stack + requirements.txt
REM ------------------------------------------------------------

set "ROCM_ROOT=%USERPROFILE%\rocm-pytorch"
if not exist "%ROCM_ROOT%" (
    mkdir "%ROCM_ROOT%" 2>nul
)

set "TAG=v6.5.0rc-pytorch-gfx110x"
set "BASE_URL=https://github.com/scottt/rocm-TheRock/releases/download/%TAG%"

set "TORCH_WHL=%ROCM_ROOT%\torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl"
set "VISION_WHL=%ROCM_ROOT%\torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl"
set "AUDIO_WHL=%ROCM_ROOT%\torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl"

if not exist "%TORCH_WHL%" (
    echo Downloading ROCm torch wheel...
    curl.exe -L "%BASE_URL%/torch-2.7.0a0+git3f903c3-cp312-cp312-win_amd64.whl" -o "%TORCH_WHL%"
)
if not exist "%VISION_WHL%" (
    echo Downloading ROCm torchvision wheel...
    curl.exe -L "%BASE_URL%/torchvision-0.22.0+9eb57cd-cp312-cp312-win_amd64.whl" -o "%VISION_WHL%"
)
if not exist "%AUDIO_WHL%" (
    echo Downloading ROCm torchaudio wheel...
    curl.exe -L "%BASE_URL%/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl" -o "%AUDIO_WHL%"
)

if not exist "%TORCH_WHL%" (
    echo ERROR: Failed to obtain torch ROCm wheel.
    goto :eof
)
if not exist "%VISION_WHL%" (
    echo ERROR: Failed to obtain torchvision ROCm wheel.
    goto :eof
)
if not exist "%AUDIO_WHL%" (
    echo ERROR: Failed to obtain torchaudio ROCm wheel.
    goto :eof
)

REM Create venv with Python 3.12 if it does not exist
if not exist "venv" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv venv
)

call "%~dp0venv\Scripts\activate.bat"

python -m ensurepip --upgrade
python -m pip install --upgrade pip

REM Install ROCm PyTorch stack from downloaded wheels
echo Installing ROCm torch/vision/audio from %ROCM_ROOT% ...
python -m pip install "%TORCH_WHL%" "%VISION_WHL%" "%AUDIO_WHL%"

REM Ensure NumPy <2 to match ROCm wheels (defensive)
python -m pip install "numpy<2"

REM Install remaining dependencies for Picture AI
python -m pip install -r requirements.txt

echo.
echo ROCm environment setup complete. To use it later, run:
echo   venv\Scripts\activate
endlocal
