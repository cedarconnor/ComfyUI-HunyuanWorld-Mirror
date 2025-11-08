# Installation Troubleshooting Guide

## Error: "ModuleNotFoundError: No module named 'plyfile'"

This error means the dependencies haven't been installed yet.

### Quick Fix (Windows)

**Option 1: Run the installer (Recommended)**
```batch
cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
install.bat
```

**Option 2: Install dependencies manually**
```batch
cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
pip install -r requirements.txt
```

### Quick Fix (Linux)

**Option 1: Run the installer (Recommended)**
```bash
cd ComfyUI/custom_nodes/ComfyUI-HunyuanWorld-Mirror
chmod +x install.sh
./install.sh
```

**Option 2: Install dependencies manually**
```bash
cd ComfyUI/custom_nodes/ComfyUI-HunyuanWorld-Mirror
pip install -r requirements.txt
```

### After Installing

1. **Restart ComfyUI completely** (close and reopen)
2. Check the console for the welcome message:
   ```
   ============================================================
    HunyuanWorld-Mirror ComfyUI Node Pack
    Version: 1.0.0
    Transform 2D images into 3D worlds
   ============================================================
   ```
3. Nodes should now appear under **"HunyuanWorld-Mirror"** category

---

## Other Common Issues

### Issue: "Python 3.10 required"
**Solution:** Install Python 3.10 from https://www.python.org/downloads/

### Issue: "CUDA not detected"
**Solution:** Install CUDA 12.x from https://developer.nvidia.com/cuda-downloads
- This is optional but recommended for GPU acceleration
- The nodes will work on CPU but will be slower

### Issue: "pip command not found"
**Solution:**
```batch
python -m pip install -r requirements.txt
```

### Issue: "Permission denied" (Linux)
**Solution:**
```bash
sudo pip install -r requirements.txt
```

### Issue: Installation script won't run
**Solution (Windows):**
- Right-click `install.bat` and select "Run as Administrator"

**Solution (Linux):**
```bash
chmod +x install.sh
./install.sh
```

---

## Verifying Installation

After installation, you can verify by checking:

1. **Check installed packages:**
   ```bash
   pip list | grep plyfile
   pip list | grep torch
   ```

2. **Check ComfyUI console** for any error messages when it starts

3. **Look for nodes** in ComfyUI under the "HunyuanWorld-Mirror" category

---

## Still Having Issues?

1. Check that you're in the correct directory
2. Make sure ComfyUI is closed during installation
3. Try restarting your computer after installation
4. Check the [GitHub Issues](https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror/issues) for similar problems

---

## Manual Dependency List

If automatic installation fails, install these manually:

```bash
pip install torch==2.4.0 torchvision==0.19.0
pip install numpy>=1.24.0,<2.0.0
pip install opencv-python Pillow imageio
pip install transformers diffusers huggingface-hub safetensors
pip install einops timm omegaconf scipy tqdm
pip install trimesh plyfile open3d
pip install matplotlib
pip install ninja jaxtyping rich
pip install gsplat
```
