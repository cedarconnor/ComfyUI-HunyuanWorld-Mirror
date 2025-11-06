@echo off
REM HunyuanWorld-Mirror ComfyUI Node Pack
REM Windows Installation Script

echo ============================================
echo HunyuanWorld-Mirror ComfyUI Installation
echo ============================================
echo.

REM Check Python version
echo Checking Python version...
python --version 2>nul | findstr "3.10" >nul
if errorlevel 1 (
    echo ERROR: Python 3.10 is required
    echo.
    echo Please install Python 3.10 from: https://www.python.org/downloads/
    echo Make sure Python is added to PATH during installation
    pause
    exit /b 1
)
echo   Python 3.10 found
echo.

REM Check CUDA
echo Checking CUDA...
nvcc --version 2>nul | findstr "12" >nul
if errorlevel 1 (
    echo WARNING: CUDA 12.x not detected
    echo   The installation may still work with other CUDA versions
    echo   or in CPU mode, but GPU acceleration is recommended
    echo.
) else (
    echo   CUDA 12.x detected
    echo.
)

REM Install core dependencies
echo Installing Python dependencies...
echo   This may take several minutes...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo.
    echo Troubleshooting:
    echo   1. Make sure you have internet connection
    echo   2. Try running as Administrator
    echo   3. Check if pip is up to date: python -m pip install --upgrade pip
    pause
    exit /b 1
)
echo.
echo   Core dependencies installed successfully
echo.

REM Install gsplat (pre-compiled wheels)
echo Installing gsplat (3D Gaussian Splatting)...
echo   Installing gsplat dependencies...
pip install ninja numpy jaxtyping rich
if errorlevel 1 (
    echo WARNING: Failed to install gsplat dependencies
)

echo   Installing gsplat from pre-compiled wheels...
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
if errorlevel 1 (
    echo.
    echo WARNING: Failed to install gsplat from pre-compiled wheels
    echo   Trying PyPI version...
    pip install gsplat
    if errorlevel 1 (
        echo.
        echo WARNING: gsplat installation failed
        echo   Gaussian splatting features may not work
        echo   You can try installing manually later: pip install gsplat
        echo.
    )
)
echo.

REM Create output directory
echo Creating output directory...
if not exist "output" mkdir output
echo   Created: ./output/
echo.

REM Installation complete
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Next Steps:
echo   1. Restart ComfyUI completely
echo   2. Nodes will appear under: HunyuanWorld-Mirror
echo   3. Check examples/ folder for workflow templates
echo.
echo System Requirements:
echo   - GPU: NVIDIA RTX 3060 or better (12GB+ VRAM)
echo   - RAM: 16GB+ system memory
echo   - CUDA: 12.1 or higher recommended
echo.
echo Troubleshooting:
echo   - If nodes don't appear, check ComfyUI console for errors
echo   - For help, visit: https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror
echo.
pause
