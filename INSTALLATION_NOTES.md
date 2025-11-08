# Installation Notes

## Handling Version Conflicts

If you see dependency conflicts during installation (like torch, torchvision, etc.), **this is usually okay**. The nodes will work with the versions you already have installed.

### Understanding the Conflicts

When you run `pip install -r requirements.txt`, you might see messages like:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
nerfstudio 0.3.4 requires timm==0.6.7, but you have timm 1.0.22 which is incompatible.
sam-2 1.0 requires torch>=2.5.1, but you have torch 2.4.0 which is incompatible.
```

**This is normal!** ComfyUI environments often have many custom nodes with conflicting requirements.

### What Actually Matters

Our node pack works with a wide range of package versions. The critical packages we need are:

1. **PyTorch** - Any version >= 2.0.0 (you probably already have this)
2. **plyfile** - For PLY file export (THIS IS THE KEY MISSING PACKAGE)
3. **open3d** - For 3D processing
4. **trimesh** - For 3D mesh handling

## Installation Options

### Option 1: Full Installation (Recommended)

This installs all dependencies, but won't downgrade existing packages:

```bash
cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
pip install -r requirements.txt
```

**Ignore the dependency conflict warnings** - they're usually harmless.

### Option 2: Minimal Installation (If you have conflicts)

This only installs the packages you probably don't have yet:

```bash
cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
pip install -r requirements-minimal.txt
```

### Option 3: Install Only What's Missing

If you want maximum control, manually install just what you need:

```bash
pip install plyfile trimesh open3d
```

Then try loading the nodes in ComfyUI. If you get import errors, install the missing package.

## After Installation

1. **Restart ComfyUI completely** (close and reopen)
2. Check the console for the welcome message:
   ```
   ============================================================
    HunyuanWorld-Mirror ComfyUI Node Pack
    Version: 1.0.0
   ============================================================
   ```
3. Look for nodes under **"HunyuanWorld-Mirror"** category

## Troubleshooting Specific Errors

### "No module named 'plyfile'"
```bash
pip install plyfile
```

### "No module named 'open3d'"
```bash
pip install open3d
```

### "No module named 'trimesh'"
```bash
pip install trimesh
```

### Nodes not appearing in ComfyUI
1. Check the console for any import errors
2. Make sure you fully restarted ComfyUI
3. Try: `pip install -r requirements-minimal.txt`

## Version Compatibility

Our nodes are tested with:
- **PyTorch:** 2.0.0 to 2.8.0+ (flexible)
- **Python:** 3.10, 3.11, 3.12
- **CUDA:** 11.8, 12.1, 12.4
- **Windows:** 10, 11
- **Linux:** Ubuntu 20.04, 22.04

But they should work with most versions in these ranges.

## Working with Existing Nodes

If you have other custom nodes installed (like nerfstudio, sam-2, etc.), you might see version conflicts. This is expected when different packages have different requirements.

**The nodes will likely still work!** Python is often more flexible than pip thinks.

### If Something Actually Breaks

1. Note which package causes the error
2. Try installing a compatible version:
   ```bash
   pip install package_name>=minimum_version
   ```
3. Open an issue on GitHub with the error message

## Clean Installation (Last Resort)

If nothing works, you can create a fresh venv:

```bash
# Create new virtual environment
python -m venv comfy_hwm_env

# Activate it (Windows)
comfy_hwm_env\Scripts\activate

# Install ComfyUI requirements first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install HWM requirements
cd ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
pip install -r requirements.txt
```

But this is usually not necessary!

## Summary

**TL;DR:**
1. Run `pip install -r requirements.txt` (or `requirements-minimal.txt`)
2. Ignore version conflict warnings
3. Restart ComfyUI
4. If you get import errors, install the specific missing package
5. Most likely just need: `pip install plyfile`
