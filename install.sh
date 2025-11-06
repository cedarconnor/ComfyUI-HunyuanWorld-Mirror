#!/bin/bash
# HunyuanWorld-Mirror ComfyUI Node Pack
# Linux Installation Script

set -e  # Exit on error

echo "============================================"
echo "HunyuanWorld-Mirror ComfyUI Installation"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
if ! python --version 2>&1 | grep -q "3.10"; then
    echo "ERROR: Python 3.10 is required"
    echo ""
    echo "Please install Python 3.10:"
    echo "  Ubuntu/Debian: sudo apt install python3.10"
    echo "  Or from: https://www.python.org/downloads/"
    exit 1
fi
echo "  ✓ Python 3.10 found"
echo ""

# Check CUDA (optional)
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "  ✓ CUDA detected: $CUDA_VERSION"
else
    echo "  ⚠ CUDA not detected"
    echo "    GPU acceleration may not be available"
    echo "    Installation will continue..."
fi
echo ""

# Install core dependencies
echo "Installing Python dependencies..."
echo "  This may take several minutes..."
echo ""
pip install -r requirements.txt || {
    echo ""
    echo "ERROR: Failed to install dependencies"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure you have internet connection"
    echo "  2. Try: pip install --upgrade pip"
    echo "  3. Check if you need sudo: sudo pip install -r requirements.txt"
    exit 1
}
echo ""
echo "  ✓ Core dependencies installed successfully"
echo ""

# Install gsplat
echo "Installing gsplat (3D Gaussian Splatting)..."
echo "  Installing gsplat dependencies..."
pip install ninja numpy jaxtyping rich || {
    echo "  ⚠ Warning: Failed to install some gsplat dependencies"
}

echo "  Installing gsplat from pre-compiled wheels..."
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124 || {
    echo ""
    echo "  ⚠ Warning: Failed to install gsplat from pre-compiled wheels"
    echo "    Trying PyPI version..."
    pip install gsplat || {
        echo ""
        echo "  ⚠ Warning: gsplat installation failed"
        echo "    Gaussian splatting features may not work"
        echo "    You can try installing manually later: pip install gsplat"
        echo ""
    }
}
echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p output
echo "  ✓ Created: ./output/"
echo ""

# Installation complete
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "Next Steps:"
echo "  1. Restart ComfyUI completely"
echo "  2. Nodes will appear under: HunyuanWorld-Mirror"
echo "  3. Check examples/ folder for workflow templates"
echo ""
echo "System Requirements:"
echo "  - GPU: NVIDIA RTX 3060 or better (12GB+ VRAM)"
echo "  - RAM: 16GB+ system memory"
echo "  - CUDA: 12.1 or higher recommended"
echo ""
echo "Troubleshooting:"
echo "  - If nodes don't appear, check ComfyUI console for errors"
echo "  - For help, visit: https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror"
echo ""
