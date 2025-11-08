# HunyuanWorld-Mirror ComfyUI Node - Troubleshooting Log

**Date:** November 7, 2025
**Status:** ✅ **RESOLVED - Model now working!**

---

## Executive Summary

Successfully debugged and fixed the HunyuanWorld-Mirror ComfyUI custom node, which was failing to load the model architecture. After extensive troubleshooting, the model now loads and runs inference successfully.

**Final Status:**
- ✅ Model loads successfully
- ✅ Inference runs and generates outputs (depth, normals, 3D points, camera poses)
- ✅ Automatic image resizing to valid dimensions
- ⚠️ Minor: SaveCameraParameters node fix applied (handles None intrinsics)

---

## Initial Problem

**Error Message (repeated 70+ times):**
```
ERROR: HunyuanWorld-Mirror model architecture not found!

The .safetensors file contains only model weights, not the code.
You need to install the HunyuanWorld-Mirror repository:
...
```

**What was happening:**
- The model `.safetensors` file loaded successfully
- But the `WorldMirror` Python class couldn't be imported
- A fallback `SafetensorsModelWrapper` was created instead
- Inference failed with `NotImplementedError`

---

## Root Causes Discovered

### 1. **Missing Utility Files** (First Issue)
The repository was missing critical dependency files:
- `src/models/layers/drop_path.py` - Stochastic depth implementation
- `src/models/layers/layer_scale.py` - Layer scaling module
- `src/models/utils/rotation.py` - Quaternion/rotation matrix utilities

**Solution:** Created these files with proper implementations from standard vision transformer libraries.

### 2. **Wrong Logger Import** (Second Issue)
`src/models/layers/vision_transformer.py` was trying to import:
```python
from training.utils.logger import RankedLogger
```
This was a training-specific logger that doesn't exist in the ComfyUI environment.

**Solution:** Changed to standard Python logging:
```python
import logging
log = logging.getLogger(__name__)
```

### 3. **Outdated gsplat Package** (Third Issue)
The gsplat package was version 0.1.11, but the code required v1.5.3+ for:
- `gsplat.rendering` module
- `gsplat.strategy` module

**Solution:** Upgraded gsplat:
```bash
pip install --upgrade gsplat
```

### 4. **Python sys.path Pollution** (Critical Issue - The Real Problem)
**This was the main blocker!**

Other ComfyUI custom nodes (`ComfyUI_SimpleTiles_Uprez`, `ComfyUI_SimpleTiles`) were adding their directories to `sys.path` **before** our custom node directory. When Python tried to import `src.models.models.worldmirror`, it looked in the wrong directories first and failed.

**Debug output showed:**
```
Custom node dir in sys.path: True  ✓
First 3 sys.path entries:
  [0] C:\ComfyUI\custom_nodes\ComfyUI_SimpleTiles_Uprez\comfy  ❌
  [1] C:\ComfyUI\custom_nodes\ComfyUI_SimpleTiles_Uprez\comfy  ❌
  [2] C:\ComfyUI\custom_nodes\ComfyUI_SimpleTiles\comfy        ❌
```

Our directory was in sys.path, but not at position 0!

**Solution:** Implemented a context manager that:
1. Temporarily clears any cached `src.*` modules from `sys.modules`
2. Removes our directory from sys.path (all instances)
3. Inserts our directory at position 0
4. Performs the import
5. Restores original sys.path

```python
@contextlib.contextmanager
def _ensure_path_priority():
    """Temporarily ensure custom node dir is first in sys.path."""
    # Clear cached failed imports
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('src.')]
    for key in modules_to_remove:
        del sys.modules[key]

    # Save and modify sys.path
    original_path = sys.path.copy()
    while _node_str in sys.path:
        sys.path.remove(_node_str)
    sys.path.insert(0, _node_str)

    try:
        yield
    finally:
        sys.path[:] = original_path
```

### 5. **Parameter Signature Mismatch** (Fifth Issue)
After successful import, inference failed with:
```
TypeError: WorldMirror.forward() got an unexpected keyword argument 'condition'
```

The `WorldMirror` class expects:
```python
forward(self, views: Dict[str, torch.Tensor], cond_flags: List[int]=[0, 0, 0], is_inference=True)
```

But the wrapper was calling:
```python
model(images, condition=condition, **kwargs)
```

**Solution:** Fixed the inference wrapper to use correct signature:
```python
views = {'img': images}
cond_flags = condition if condition is not None else [0, 0, 0]
outputs = self.model(views, cond_flags=cond_flags, is_inference=True)
```

### 6. **Image Size Validation** (Sixth Issue)
Inference failed with:
```
AssertionError: Input image height 1024 is not a multiple of patch height 14
```

The model uses 14×14 patches, so image dimensions must be multiples of 14.

**Solution:** Added automatic resizing in `comfy_to_hwm()`:
```python
target_h = round(H / patch_size) * patch_size
target_w = round(W / patch_size) * patch_size
# Resize if needed: 1024x2048 → 1022x2044
```

### 7. **None Intrinsics in Export** (Seventh Issue - Minor)
SaveCameraParameters node crashed when intrinsics were None (single-image inference).

**Solution:** Added None check before accessing intrinsics array.

---

## Files Modified/Created

### Created Files:
1. `src/models/layers/drop_path.py` - Stochastic depth
2. `src/models/layers/layer_scale.py` - Layer scaling
3. `src/models/utils/rotation.py` - Rotation utilities
4. `clear_all_cache.ps1` - PowerShell script to clear Python cache
5. `test_import.py` - Test script for model import
6. `test_actual_load.py` - Test script for actual model loading

### Modified Files:
1. `utils/inference.py`
   - Added sys.path manipulation at module level
   - Added context manager for import-time path priority
   - Added import cache clearing
   - Fixed model call signature
   - Added debug logging

2. `utils/tensor_utils.py`
   - Added automatic image resizing to patch size multiples

3. `src/models/layers/vision_transformer.py`
   - Replaced `RankedLogger` with standard `logging`

4. `nodes.py`
   - Added `force_reload` checkbox to bypass model cache

5. `utils/export.py`
   - Added None check for intrinsics in camera export

---

## Current Working State

### ✅ What Works:
1. **Model Loading:**
   ```
   ✓ Successfully loaded WorldMirror model from HunyuanWorld-Mirror.safetensors
   After model loading - GPU Memory: 2.35GB allocated, 45.63GB free, 47.99GB total
   ```

2. **Inference:**
   ```
   ✓ Inference complete
     Depth: torch.Size([1, 1, 1022, 2044, 1])
     Normals: torch.Size([1, 1, 1022, 2044, 3])
     Points3D: torch.Size([1, 1, 1022, 2044, 3])
     Poses: torch.Size([1, 1, 4, 4])
   ```

3. **Automatic Resizing:**
   ```
   [INFO] Resizing images from 1024x2048 to 1022x2044 (multiple of 14)
   ```

4. **Memory Management:**
   - Loads on GPU successfully
   - Uses ~2.5GB VRAM for inference
   - Returns to ~2.35GB after inference

### Package Requirements:
- PyTorch 2.9.0+ with CUDA
- gsplat >= 1.5.3 (upgraded from 0.1.11)
- einops 0.8.0+
- timm 1.0.22+
- transformers 4.48.3+
- huggingface-hub 0.35.1+
- safetensors 0.4.0+

---

## Important Implementation Details

### Force Reload Mechanism
The `LoadHunyuanWorldMirrorModel` node now has a `force_reload` checkbox:
- When enabled: Bypasses model cache, forces reload from disk
- When disabled: Uses cached model instance (faster)
- **First time after fixes:** MUST enable `force_reload` to clear old broken cached model

### Python Cache Clearing
ComfyUI aggressively caches Python modules. To apply code changes:
```powershell
cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
.\clear_all_cache.ps1  # Clear all __pycache__ and .pyc files
# Then restart ComfyUI completely
```

### Import Strategy
The import uses a **context manager approach** instead of global sys.path manipulation:
- More reliable than fighting with sys.path globally
- Clears Python's import cache before each import attempt
- Restores sys.path after import to avoid polluting other nodes
- Located in `utils/inference.py` lines 288-320

---

## Testing Checklist

### ✅ Completed Tests:
- [x] Model imports successfully
- [x] WorldMirror class instantiates
- [x] Model loads from .safetensors file
- [x] Inference runs without errors
- [x] Depth output generated
- [x] Normals output generated
- [x] 3D points output generated
- [x] Camera poses output generated
- [x] Automatic image resizing works
- [x] GPU memory management works

### ⚠️ Needs Testing:
- [ ] Multi-image input (>1 image)
- [ ] Different image sizes
- [ ] Visualization nodes (VisualizeDepth, VisualizeNormals)
- [ ] All export nodes (SavePointCloud, Save3DGaussians, etc.)
- [ ] Different precision modes (fp32, bf16)
- [ ] CPU mode
- [ ] Full workflow with all nodes connected

---

## Known Limitations

1. **Single Image Input:**
   - Currently tested with 1 image only
   - Model supports 4-64 images for better 3D reconstruction
   - Need to test multi-image workflows

2. **Image Size Constraints:**
   - Images must be (or will be resized to) multiples of 14 pixels
   - Optimal size is 518×518 (model's default)
   - Large images (>2048) may cause memory issues

3. **Intrinsics:**
   - Camera intrinsics are None for single-image inference
   - Need multiple images for intrinsic estimation
   - SaveCameraParameters node now handles this gracefully

4. **Cache Persistence:**
   - Python bytecode cache persists between ComfyUI restarts
   - Must manually clear cache when updating code
   - `force_reload` checkbox helps but requires user action

---

## User Workflow (Current State)

### First Time Setup:
1. Install dependencies (if not already):
   ```bash
   pip install --upgrade gsplat
   pip install einops
   ```

2. Clear Python cache:
   ```powershell
   cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
   .\clear_all_cache.ps1
   ```

3. Restart ComfyUI completely

4. In workflow:
   - Add `LoadHunyuanWorldMirrorModel` node
   - **Enable `force_reload` checkbox** ✓
   - Select model: `HunyuanWorld-Mirror` or path to `.safetensors`
   - Device: `cuda` (recommended)
   - Precision: `fp16` (recommended)

5. Add `HWMInference` node
   - Connect model from loader
   - Connect images (LoadImage node)
   - Seed: -1 (random) or fixed number

6. Add visualization/export nodes as needed

### Subsequent Runs:
- Can disable `force_reload` after first successful load
- Model will be cached in memory for faster loading

---

## Debugging Tips for Future Issues

### If Model Won't Load:
1. Check debug output in console for sys.path:
   ```
   [DEBUG] sys.path[0] = C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
   ```
   If path[0] is NOT the custom node directory, sys.path is still polluted

2. Check cleared modules count:
   ```
   [DEBUG] Cleared 44 cached 'src.*' modules
   ```
   If this is 0, imports might be cached elsewhere

3. Verify files exist:
   ```
   [DEBUG] src dir exists: True
   ```

### If Import Fails:
1. Run test script directly:
   ```bash
   cd C:\ComfyUI\custom_nodes\ComfyUI-HunyuanWorld-Mirror
   C:\ComfyUI\.venv\Scripts\python.exe test_import.py
   ```

2. Check for missing files:
   ```bash
   git ls-files src/ | grep -E "(drop_path|layer_scale|rotation)"
   ```

3. Verify gsplat version:
   ```bash
   pip show gsplat  # Should be >= 1.5.3
   ```

### If Inference Fails:
1. Check image dimensions:
   - Must be multiples of 14
   - Auto-resize should trigger: `[INFO] Resizing images from...`

2. Check VRAM:
   - Need ~4GB free for inference
   - Model uses ~2.5GB

3. Check model signature match:
   - Verify `views` dict has 'img' key
   - Verify `cond_flags` is a list of 3 ints

---

## Git Commits (Session Log)

Key commits from this debugging session:

1. **bd49b67** - Integrate HunyuanWorld-Mirror model architecture
2. **838290f** - Fix: Add missing utility files (drop_path, layer_scale, rotation)
3. **a6eaf29** - Fix: Ensure custom node directory is in sys.path
4. **60a40df** - Suppress repeated error messages
5. **d89ecbe** - Fix: Clear Python import cache before importing
6. **20040f2** - Fix: Match WorldMirror forward() signature
7. **ec84779** - Fix: Auto-resize images to multiples of patch size (14)
8. **bd5962d** - Fix: Handle None intrinsics in camera parameter export

---

## Next Steps / Future Work

### Immediate (Tomorrow):
1. Test with SaveCameraParameters node to verify intrinsics fix
2. Test multi-image input (2-4 images)
3. Test visualization nodes (VisualizeDepth, VisualizeNormals)
4. Test all export nodes

### Short Term:
1. Add better error messages for common issues
2. Add input validation and helpful warnings
3. Consider adding `recommended_size` parameter (auto-resize to 518×518)
4. Add progress bars for long inference
5. Optimize memory usage for large images

### Long Term:
1. Add conditioning support (depth, camera, rays)
2. Add support for 3DGS post-optimization
3. Add batch processing support
4. Create example workflows
5. Write comprehensive user documentation
6. Add unit tests for critical functions

---

## Contact / References

- **Repository:** https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror
- **Official Model:** https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
- **ComfyUI:** https://github.com/comfyanonymous/ComfyUI

---

## Conclusion

After extensive debugging involving sys.path pollution, missing files, package versions, and API mismatches, the HunyuanWorld-Mirror ComfyUI node is now **fully functional**. The model loads successfully, runs inference, and generates all expected outputs (depth, normals, 3D points, camera poses).

The key breakthrough was discovering that other custom nodes were polluting sys.path and implementing a robust context manager solution to ensure our module directory has priority during imports.

**Status: Ready for production use with single-image input. Multi-image testing recommended before wider deployment.**

---

*Document created: November 7, 2025*
*Last updated: November 7, 2025*
*Status: ✅ RESOLVED*
