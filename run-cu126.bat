@echo off
setlocal

REM Check if venv directory exists
if exist venv (
    echo Activating virtual environment...
) else (
    echo venv directory not found. Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        exit /b %errorlevel%
    )
    echo Installing requirements...
    .\venv\Scripts\python -m pip install Flask
    .\venv\Scripts\python -m pip install diffusers==0.32.2
    .\venv\Scripts\python -m pip install Pillow
    .\venv\Scripts\python -m pip install transformers==4.48.1
    .\venv\Scripts\python -m pip install opencv-python==4.10.0.84
    .\venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    .\venv\Scripts\python -m pip install compel
    .\venv\Scripts\python -m pip install accelerate

    if %errorlevel% neq 0 (
        echo Failed to install requirements.
        exit /b %errorlevel%
    )
)

REM Check for CUDA
echo Checking for CUDA...
.\venv\Scripts\python -c "import torch; print(torch.cuda.is_available())" > cuda_check.txt
set /p cuda_available=<cuda_check.txt
del cuda_check.txt

if "%cuda_available%" == "True" (
    echo CUDA is available.
    .\venv\Scripts\python app.py
) else (
    echo CUDA is not available. CUDA is required to run this application.
    echo https://developer.nvidia.com/cuda-downloads to download CUDA.
)

endlocal