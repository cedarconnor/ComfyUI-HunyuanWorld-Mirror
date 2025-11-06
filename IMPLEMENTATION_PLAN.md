# ComfyUI-HunyuanWorld-Mirror Implementation Plan
## Streamlined Development Approach

**Version:** 2.0 (Optimized)
**Date:** November 6, 2025
**Focus:** Minimal nodes, maximum functionality

---

## Overview

This plan streamlines the original design by:
1. **Reducing node count** from 15+ to 8 core nodes
2. **Leveraging ComfyUI defaults** for image loading and previewing
3. **Simplifying installation** with automated scripts
4. **Focusing on core functionality** first, advanced features later

---

## Phase 1: Core Implementation (Week 1-2)

### Project Structure

```
ComfyUI-HunyuanWorld-Mirror/
├── __init__.py                  # Node registration
├── nodes.py                     # All node implementations in one file
├── utils/
│   ├── __init__.py
│   ├── tensor_utils.py         # Format conversions
│   ├── inference.py            # Model inference wrapper
│   ├── export.py               # Export utilities
│   └── memory.py               # Memory management
├── requirements.txt            # Dependencies
├── install.bat                 # Windows installer
├── install.sh                  # Linux installer
├── README.md                   # User documentation
└── examples/
    ├── basic_workflow.json     # Simple reconstruction
    └── advanced_workflow.json  # With all features

```

### 8 Core Nodes

#### 1. LoadHunyuanWorldMirrorModel
```python
INPUTS:
  - model_name: STRING (default: "tencent/HunyuanWorld-Mirror")
  - device: ["auto", "cuda", "cpu"]
  - precision: ["fp32", "fp16", "bf16"]

OUTPUTS:
  - MODEL: Loaded model instance

FEATURES:
  - Automatic model caching
  - HuggingFace Hub integration
  - Memory usage reporting
```

#### 2. HWMInference
```python
INPUTS:
  - model: MODEL
  - images: IMAGE (from ComfyUI LoadImage/ImageBatch)
  - seed: INT (default: -1)

OUTPUTS:
  - depth: DEPTH (raw depth maps)
  - normals: NORMALS (surface normals)
  - points3d: POINTS3D (3D point cloud)
  - camera_poses: POSES (camera-to-world matrices)
  - camera_intrinsics: INTRINSICS (camera K matrices)
  - gaussians: GAUSSIANS (3D Gaussian splat parameters)

FEATURES:
  - Single forward pass for all outputs
  - Automatic format conversion (ComfyUI ↔ HWM)
  - Memory-efficient processing
  - Progress reporting
```

#### 3. VisualizeDepth
```python
INPUTS:
  - depth: DEPTH
  - colormap: ["viridis", "plasma", "turbo", "gray"]
  - normalize: BOOLEAN (default: True)

OUTPUTS:
  - IMAGE: Colorized depth map (can connect to PreviewImage)

FEATURES:
  - Multiple colormaps
  - Auto-normalization
  - ComfyUI-compatible image output
```

#### 4. VisualizeNormals
```python
INPUTS:
  - normals: NORMALS

OUTPUTS:
  - IMAGE: RGB visualization (X→R, Y→G, Z→B)

FEATURES:
  - Standard normal visualization
  - Direct preview support
```

#### 5. SavePointCloud
```python
INPUTS:
  - points3d: POINTS3D
  - colors: IMAGE (optional, from source images)
  - normals: NORMALS (optional)
  - filepath: STRING
  - format: ["ply", "pcd", "obj", "xyz"]

OUTPUTS:
  - filepath: STRING (confirmation)

FEATURES:
  - Multiple formats
  - Binary and ASCII PLY
  - Optional attributes (color, normals)
```

#### 6. Save3DGaussians
```python
INPUTS:
  - gaussians: GAUSSIANS
  - filepath: STRING
  - include_sh: BOOLEAN (default: False)

OUTPUTS:
  - filepath: STRING

FEATURES:
  - Standard 3DGS PLY format
  - Compatible with SuperSplat, gsplat viewers
  - Optional spherical harmonics
```

#### 7. SaveDepthMap
```python
INPUTS:
  - depth: DEPTH
  - filepath: STRING
  - format: ["npy", "exr", "pfm", "png16"]

OUTPUTS:
  - filepath: STRING

FEATURES:
  - Multiple precision formats
  - Metric depth preservation
```

#### 8. SaveCameraParams
```python
INPUTS:
  - camera_poses: POSES
  - camera_intrinsics: INTRINSICS
  - filepath: STRING
  - format: ["json", "colmap", "npy"]

OUTPUTS:
  - filepath: STRING

FEATURES:
  - Human-readable JSON
  - COLMAP compatibility
  - NumPy arrays
```

---

## Leveraging ComfyUI Built-in Nodes

### Image Loading
**Instead of custom PrepareImageSequence:**
1. Use `LoadImage` (built-in) → loads images
2. Use `ImageBatch` (built-in) → creates sequences
3. Use `ImageScale` (built-in) → resize to 518x518 if needed

### Previewing
**Instead of custom preview nodes:**
1. VisualizeDepth → `PreviewImage` (built-in)
2. VisualizeNormals → `PreviewImage` (built-in)
3. Connect IMAGE outputs directly to preview

### String Inputs
**Use ComfyUI's native STRING widgets for:**
- File paths
- Model names
- Output locations

---

## Installation Strategy

### Windows Installation (`install.bat`)

```batch
@echo off
echo ============================================
echo HunyuanWorld-Mirror ComfyUI Installation
echo ============================================

REM Check Python 3.10
python --version | findstr "3.10" >nul
if errorlevel 1 (
    echo ERROR: Python 3.10 required
    pause && exit /b 1
)

REM Check CUDA
nvcc --version | findstr "12" >nul
if errorlevel 1 (
    echo WARNING: CUDA 12.x recommended
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt || (
    echo ERROR: Failed to install dependencies
    pause && exit /b 1
)

REM Install gsplat (pre-compiled)
echo Installing gsplat...
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo Restart ComfyUI to load nodes
pause
```

### Linux Installation (`install.sh`)

```bash
#!/bin/bash
set -e

echo "============================================"
echo "HunyuanWorld-Mirror ComfyUI Installation"
echo "============================================"

# Check Python
if ! python --version | grep -q "3.10"; then
    echo "ERROR: Python 3.10 required"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install gsplat
echo "Installing gsplat..."
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
```

### Dependencies (`requirements.txt`)

```
# Core
torch==2.4.0
torchvision==0.19.0
numpy>=1.24.0,<2.0.0

# Image processing
opencv-python>=4.8.0
Pillow>=10.0.0
imageio>=2.31.0

# Deep learning
einops>=0.7.0
timm>=0.9.0
transformers>=4.35.0
diffusers>=0.24.0

# Model loading
huggingface-hub>=0.19.0
safetensors>=0.4.0

# 3D export
trimesh>=4.0.0
plyfile>=1.0.0
open3d>=0.18.0

# Utilities
tqdm>=4.66.0
omegaconf>=2.3.0
scipy>=1.11.0

# gsplat dependencies (installed separately)
ninja
jaxtyping
rich
```

---

## Development Timeline

### Week 1: Foundation
- [x] Project structure setup
- [ ] `utils/tensor_utils.py` - Format conversions
- [ ] `utils/inference.py` - Model wrapper
- [ ] `utils/memory.py` - Memory management
- [ ] Node 1: LoadHunyuanWorldMirrorModel
- [ ] Node 2: HWMInference

### Week 2: Outputs & Testing
- [ ] `utils/export.py` - Export functions
- [ ] Nodes 3-8: All output/save nodes
- [ ] `__init__.py` - Node registration
- [ ] Basic workflow example
- [ ] Installation script testing

### Week 3: Documentation & Polish
- [ ] README.md (comprehensive)
- [ ] Example workflows
- [ ] Troubleshooting guide
- [ ] Video tutorial planning

---

## Key Implementation Details

### Tensor Format Conversion

```python
# utils/tensor_utils.py

def comfy_to_hwm(images: torch.Tensor) -> torch.Tensor:
    """
    ComfyUI: [B, H, W, C] in [0,1]
    HWM:     [1, N, 3, H, W] in [0,1]
    """
    # [B, H, W, C] -> [B, C, H, W] -> [1, B, C, H, W]
    return images.permute(0, 3, 1, 2).unsqueeze(0)

def hwm_to_comfy(images: torch.Tensor) -> torch.Tensor:
    """
    HWM:     [1, N, 3, H, W]
    ComfyUI: [B, H, W, C]
    """
    # [1, N, 3, H, W] -> [N, 3, H, W] -> [N, H, W, 3]
    return images.squeeze(0).permute(0, 2, 3, 1)
```

### Model Loading with Cache

```python
# utils/inference.py

class ModelCache:
    _cache = {}

    @classmethod
    def get(cls, key):
        return cls._cache.get(key)

    @classmethod
    def set(cls, key, model):
        cls._cache[key] = model
```

### Memory Management

```python
# utils/memory.py

def clear_gpu_memory():
    """Clear CUDA cache after inference."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def estimate_memory(num_frames, height, width, precision="fp32"):
    """Estimate VRAM needed."""
    bytes_per_pixel = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]
    # Input + outputs (rough estimate)
    total = num_frames * height * width * 3 * bytes_per_pixel * 6
    return total / (1024**3)  # GB
```

---

## Example Workflow

### Basic Reconstruction Workflow

```
LoadImage → ImageBatch → HWMInference → Outputs:
                            ↓
              ┌─────────────┼─────────────┬─────────────┐
              ↓             ↓             ↓             ↓
        VisualizeDepth  VisualizeNormals  SavePointCloud  Save3DGaussians
              ↓             ↓
        PreviewImage    PreviewImage
```

**Nodes Used:**
1. LoadImage (built-in)
2. ImageBatch (built-in)
3. LoadHunyuanWorldMirrorModel (custom)
4. HWMInference (custom)
5. VisualizeDepth (custom)
6. VisualizeNormals (custom)
7. SavePointCloud (custom)
8. Save3DGaussians (custom)
9. PreviewImage (built-in) x2

**Total: 6 custom nodes in typical workflow**

---

## Testing Strategy

### Unit Tests
```python
# tests/test_tensor_utils.py
def test_comfy_to_hwm_conversion()
def test_hwm_to_comfy_conversion()
def test_round_trip_conversion()
```

### Integration Tests
```python
# tests/test_integration.py
def test_end_to_end_inference()
def test_export_formats()
def test_memory_cleanup()
```

### Manual Testing Checklist
- [ ] Windows installation on clean system
- [ ] Linux installation on clean system
- [ ] Model loading and caching
- [ ] Inference with various image counts (4, 8, 16)
- [ ] All export formats
- [ ] Memory usage stays < 12GB
- [ ] Preview images display correctly

---

## Documentation Requirements

### README.md Must Include:
1. Project overview and features
2. System requirements
3. Installation steps (Windows/Linux)
4. Quick start guide
5. Basic workflow example
6. Output format specifications
7. Troubleshooting section
8. Links to examples

### Example Workflows to Provide:
1. `basic_workflow.json` - Simple image to 3D
2. `advanced_workflow.json` - All features demo

---

## Success Criteria

### Installation
- ✅ One-command install on Windows/Linux
- ✅ Clear error messages for missing prerequisites
- ✅ Works with Python 3.10 + CUDA 12.x
- ✅ Installation completes in < 10 minutes

### Functionality
- ✅ All 8 core nodes working
- ✅ Handles 4-64 frame sequences
- ✅ All output formats valid
- ✅ Memory usage < 12GB for typical scenes

### User Experience
- ✅ Nodes appear in ComfyUI menu under "HunyuanWorld-Mirror"
- ✅ Clear node descriptions and tooltips
- ✅ Example workflows load and run successfully
- ✅ Documentation answers common questions

### Performance Targets
- ✅ 4 frames: < 5 seconds
- ✅ 16 frames: < 15 seconds
- ✅ Model load: < 30 seconds (first time)
- ✅ Memory: < 10GB peak for standard scenes

---

## Phase 2: Advanced Features (Future)

### Additional Nodes (Optional):
1. **LoadCameraPoses** - Load pose priors
2. **LoadDepthMaps** - Load depth priors
3. **LoadIntrinsics** - Load intrinsic priors
4. **RefineGaussians** - Optimization loop
5. **RenderGaussianView** - Novel view synthesis

### Enhanced Features:
- Batch processing multiple sequences
- TensorRT acceleration
- Quality presets (fast/balanced/quality)
- Mesh reconstruction from point clouds
- Video-to-3D conversion

---

## Conclusion

This streamlined plan:
- **Reduces complexity** from 15+ nodes to 8 core nodes
- **Leverages ComfyUI** built-in nodes for common tasks
- **Focuses on core functionality** needed by 90% of users
- **Provides clear path** for future enhancements
- **Ensures easy installation** with automated scripts

**Next Steps:**
1. Implement `utils/` modules (tensor, inference, export, memory)
2. Implement 8 core nodes in `nodes.py`
3. Create `__init__.py` registration
4. Write installation scripts
5. Create example workflows
6. Write comprehensive README

**Estimated Time to MVP:** 2-3 weeks
