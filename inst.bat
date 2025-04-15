@echo off
REM inst.bat
REM This script installs all required dependencies for the polygon color prediction project.
REM It installs:
REM - PyTorch (CPU version by default)
REM - numpy (for numerical computations)
REM - pillow (for image processing - PIL fork)
REM - CLIP from OpenAI's GitHub repository

REM Upgrade pip to the latest version (optional)
python -m pip install --upgrade pip

REM Install PyTorch (CPU version) along with torchvision and torchaudio
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

REM Install numpy
python -m pip install numpy

REM Install pillow
python -m pip install pillow

REM Install CLIP from OpenAI's GitHub repository
python -m pip install git+https://github.com/openai/CLIP.git

echo All dependencies have been successfully installed!
pause
