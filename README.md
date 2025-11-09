# ComfyUI-HunyuanWorld-Mirror

Transform 2D images into 3D worlds using Tencent's HunyuanWorld-Mirror model directly in ComfyUI.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-downloads)

![HunyuanWorld-Mirror Demo](docs/demo.gif)

---

## ✨ Features

- **🎯 Single-Pass 3D Reconstruction** - Generate point clouds, depth maps, normals, camera parameters, and 3D Gaussians in one forward pass
- **🎨 11 Specialized Nodes** - Complete pipeline from preprocessing to interactive 3D visualization
- **🔧 ComfyUI Native** - Seamless integration with existing image workflows
- **📦 Multiple Export Formats** - PLY, OBJ, XYZ for point clouds; NPY, EXR, PFM for depth; JSON, COLMAP for cameras
- **✨ 3D Gaussian Splatting** - Export to standard 3DGS format for novel view synthesis
- **🖥️ Interactive 3D Viewers** - Built-in viewers for Gaussian splats and point clouds with orbit controls
- **💾 Memory Efficient** - Automatic batch processing handles sequences of any length
- **🚀 Easy Installation** - One-command setup for Windows and Linux
- **🎛️ Confidence Filtering** - Remove low-quality points based on model confidence
- **📏 Automatic Resizing** - Smart preprocessing handles any input resolution

---

## 📋 What is HunyuanWorld-Mirror?

HunyuanWorld-Mirror is a universal 3D world reconstruction model developed by Tencent that:
- Takes a sequence of images (1-100+ frames) as input
- Predicts comprehensive 3D geometry in a single forward pass
- Generates multiple 3D representations simultaneously:
  - **3D Point Clouds** with colors and normals
  - **3D Gaussian Splats** for novel view synthesis
  - **Depth Maps** for each view
  - **Surface Normals** for lighting/rendering
  - **Camera Parameters** (poses and intrinsics)
- Supports optional geometric priors (camera poses, depth, intrinsics)

This ComfyUI node pack brings these capabilities to your image generation workflows.

---

## 🎬 Optimal Input Specifications

### Image Formats
**Supported Formats:**
- JPEG/JPG (.jpg, .jpeg) - ✅ Recommended for photos
- PNG (.png) - ✅ Recommended for renders/synthetic
- WebP (.webp) - ✅ Good compression
- BMP, TIFF, TGA - ✅ Supported

**Resolution Guidelines:**
- **Input:** Any resolution (automatically preprocessed)
- **Internal Processing:** Resized to multiples of 14 pixels
- **Default Target:** 518×518 (optimal quality/speed balance)
- **Supported Range:** 224-1024 pixels (steps of 14)
  - 224×224: Fastest, lower quality
  - 518×518: **Recommended** - best balance
  - 644×644: Higher quality, slower
  - 1024×1024: Maximum quality, slowest

**Color Space:**
- RGB images (3 channels)
- Automatic normalization to [0, 1] range

### Frame Count

| Use Case | Frame Count | Memory Usage | Notes |
|----------|-------------|--------------|-------|
| Single image | 1 frame | ~4GB | Useful for depth/normals only |
| Quick test | 4-8 frames | ~6-8GB | Fast inference |
| **Recommended** | **8-24 frames** | **8-12GB** | **Best quality/speed** |
| High coverage | 24-48 frames | 12-16GB | Better 3D reconstruction |
| Maximum quality | 48-100+ frames | 16GB+ | Use batch processing |

**Important:** No hard upper limit! Batch processing (default: 16 frames/batch) automatically handles any sequence length.

### Camera Motion

**Best Results:**
- Forward motion (moving toward/away from subject)
- Orbital motion (circling around subject)
- Mixed motion (combination of forward + orbital)
- Consistent overlap between consecutive frames (50-70%)

**Avoid:**
- Pure rotation without translation (no parallax)
- Extreme motion blur
- Completely unrelated scenes

### Image Quality

**Recommended:**
- Well-lit scenes with good contrast
- Sharp focus (avoid blur)
- Consistent lighting across sequence
- Minimal motion blur
- Similar exposure across frames

---

## 📥 Installation

### Prerequisites

- **Python 3.10** (required)
- **CUDA 12.x** (recommended for GPU acceleration)
- **ComfyUI** installed and working
- **12GB+ VRAM** recommended (RTX 3060 or better)
- **16GB+ RAM** system memory

### Windows Installation

1. **Navigate to your ComfyUI custom nodes folder:**
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **Clone this repository:**
   ```bash
   git clone https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror.git
   cd ComfyUI-HunyuanWorld-Mirror
   ```

3. **Run the installation script:**
   ```bash
   install.bat
   ```

4. **Restart ComfyUI**

### Linux Installation

1. **Navigate to your ComfyUI custom nodes folder:**
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **Clone this repository:**
   ```bash
   git clone https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror.git
   cd ComfyUI-HunyuanWorld-Mirror
   ```

3. **Run the installation script:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. **Restart ComfyUI**

### Model Installation

The HunyuanWorld-Mirror model files need to be placed in ComfyUI's models directory:

1. **Create the models directory** (if it doesn't exist):
   ```bash
   mkdir -p ComfyUI/models/HunyuanWorld-Mirror
   ```

2. **Place your model file:**
   - Supported formats: `.safetensors`, `.pt`, `.pth`
   - Recommended location: `ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors`
   - Download from: [HuggingFace](https://huggingface.co/tencent/HunyuanWorld-Mirror) (if available)

3. **Using the model in ComfyUI:**
   - In the `LoadHunyuanWorldMirrorModel` node, you can use:
     - **Filename only**: `HunyuanWorld-Mirror.safetensors`
     - **Full path**: `C:/ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors`

---

## 🚀 Quick Start

### Basic Workflow

1. **Load Images:**
   - Use `LoadImage` (built-in) for individual images
   - Use `LoadImageBatch` or `VHS VideoLoader` for sequences

2. **(Optional) Preprocess Images:**
   - Add `PreprocessImagesForHWM` node
   - Choose crop/pad strategy
   - Set target size (default 518)

3. **Load Model:**
   - Add `LoadHunyuanWorldMirrorModel` node
   - Select device (auto/cuda/cpu) and precision (fp16 recommended)

4. **Run Inference:**
   - Add `HWMInference` node
   - Connect model and images
   - Adjust batch_size if needed (default: 16)

5. **Visualize Results:**
   - Connect `depth` → `VisualizeDepth` → `PreviewImage`
   - Connect `normals` → `VisualizeNormals` → `PreviewImage`

6. **View 3D Data Interactively:**
   - Connect `gaussians` filepath → `Preview3DGS` (interactive viewer)
   - Connect `points3d` filepath → `PreviewPointCloud` (interactive viewer)

7. **Export 3D Data:**
   - Connect `points3d` → `SavePointCloud`
   - Connect `gaussians` → `Save3DGaussians`

---

## 🎯 Node Reference

### 1. PreprocessImagesForHWM

Preprocess images to meet model requirements with professional crop/pad strategies.

**Inputs:**
- `images` (IMAGE): Input image sequence
- `strategy` (COMBO): ["crop", "pad"] - Preprocessing method
- `target_size` (INT): Target size in pixels (224-1024, step: 14, default: 518)

**Outputs:**
- `IMAGE`: Preprocessed images ready for inference

**Notes:**
- **Crop**: Resizes and center-crops to target size (maintains aspect ratio)
- **Pad**: Scales and pads with white to target size (preserves full content)
- Images automatically adjusted to multiples of 14 pixels

---

### 2. LoadHunyuanWorldMirrorModel

Load the HunyuanWorld-Mirror model with automatic caching.

**Inputs:**
- `model_name` (STRING): Model path or filename
- `device` (COMBO): ["auto", "cuda", "cpu"]
- `precision` (COMBO): ["fp32", "fp16", "bf16"]

**Outputs:**
- `MODEL`: Loaded model instance

**Notes:**
- Use `fp16` to reduce memory usage by 50%
- Model is cached - subsequent loads are instant

---

### 3. HWMInference

Main inference node that generates all 3D predictions in one pass.

**Inputs:**
- `model` (MODEL): Loaded model
- `images` (IMAGE): Image sequence (1-100+ frames)
- `seed` (INT): Random seed (-1 for random)
- `batch_size` (INT): Frames per batch (default: 16)

**Outputs:**
- `depth` (DEPTH): Multi-view depth maps
- `normals` (NORMALS): Surface normals
- `points3d` (POINTS3D): 3D point cloud
- `camera_poses` (POSES): Camera-to-world matrices
- `camera_intrinsics` (INTRINSICS): Intrinsic matrices
- `gaussians` (GAUSSIANS): 3D Gaussian splat parameters

**Batch Processing:**
- **8-16 batch size**: Good for 50-100 frames on 12GB VRAM
- **32+ batch size**: Better for small sequences on high VRAM
- Automatically handles any sequence length

---

### 4. VisualizeDepth

Convert depth maps to colorized visualizations.

**Inputs:**
- `depth` (DEPTH): Depth data from HWMInference
- `colormap` (COMBO): ["viridis", "plasma", "turbo", "gray", "magma"]
- `normalize` (BOOLEAN): Auto-scale depth range

**Outputs:**
- `IMAGE`: Colorized depth map

---

### 5. VisualizeNormals

Convert surface normals to RGB visualization (X→R, Y→G, Z→B).

**Inputs:**
- `normals` (NORMALS): Normal data from HWMInference

**Outputs:**
- `IMAGE`: RGB visualization

---

### 6. SavePointCloud

Export 3D point cloud to standard formats.

**Inputs:**
- `points3d` (POINTS3D): Point cloud data
- `filepath` (STRING): Output file path
- `format` (COMBO): ["ply", "obj", "xyz"]
- `confidence_threshold` (FLOAT): Filter low-confidence points (0-100 percentile)
- `colors` (IMAGE, optional): Point colors from source images
- `normals` (NORMALS, optional): Surface normals
- `confidence` (optional): Confidence values from model

**Outputs:**
- `filepath` (STRING): Saved file location

**Supported Formats:**
- **PLY**: Binary/ASCII, supports colors and normals (recommended)
- **OBJ**: Wavefront OBJ (points only)
- **XYZ**: Simple ASCII format

**Confidence Filtering:**
- `0`: Keep all points
- `50`: Keep top 50% most confident points
- `95`: Keep only very confident points (less noise)

**Note:** Colors and normals are automatically resized to match point cloud dimensions.

---

### 7. Save3DGaussians

Export 3D Gaussian Splatting representation.

**Inputs:**
- `gaussians` (GAUSSIANS): Gaussian data from HWMInference
- `filepath` (STRING): Output PLY file path
- `filter_scale_percentile` (FLOAT): Remove outlier Gaussians (default: 95.0)
- `include_sh` (BOOLEAN): Include spherical harmonics

**Outputs:**
- `filepath` (STRING): Saved file location

**Notes:**
- Standard 3DGS PLY format
- Compatible with SuperSplat, gsplat viewers
- Scale filtering removes overly large Gaussians (outliers)

---

### 8. SaveDepthMap

Export depth maps in various precision formats.

**Inputs:**
- `depth` (DEPTH): Depth data from HWMInference
- `filepath` (STRING): Output file path
- `format` (COMBO): ["npy", "exr", "pfm", "png16"]

**Outputs:**
- `filepath` (STRING): Saved file location

**Supported Formats:**
- **NPY**: NumPy array (full precision)
- **EXR**: OpenEXR (high dynamic range)
- **PFM**: Portable Float Map
- **PNG16**: 16-bit PNG (compressed)

---

### 9. SaveCameraParams

Export camera parameters for 3D reconstruction tools.

**Inputs:**
- `camera_poses` (POSES): Camera poses from HWMInference
- `camera_intrinsics` (INTRINSICS): Intrinsics from HWMInference
- `filepath` (STRING): Output file path
- `format` (COMBO): ["json", "colmap", "npy"]

**Outputs:**
- `filepath` (STRING): Saved file location

**Supported Formats:**
- **JSON**: Human-readable with all parameters
- **COLMAP**: Binary format for COLMAP/MVS tools
- **NPY**: NumPy arrays (separate files)

---

### 10. Preview3DGS

Interactive 3D Gaussian Splatting viewer with orbit controls.

**Inputs:**
- `gs_file_path` (STRING): Path to Gaussian splat PLY file

**Features:**
- Orbit, pan, zoom controls
- Real-time rendering
- Embedded in ComfyUI node
- Automatically updates when new file saved

---

### 11. PreviewPointCloud

Interactive point cloud viewer with orbit controls.

**Inputs:**
- `pointcloud_file_path` (STRING): Path to point cloud PLY file

**Features:**
- Orbit, pan, zoom controls
- Colored point cloud display
- Real-time rendering
- Embedded in ComfyUI node

---

## 💻 System Requirements

### Minimum Requirements

- **GPU**: NVIDIA RTX 3060 (12GB VRAM)
- **RAM**: 16GB system memory
- **Python**: 3.10
- **CUDA**: 12.1 or higher
- **Storage**: 10GB for models and cache

### Recommended Specifications

- **GPU**: NVIDIA RTX 4080 or better (16GB+ VRAM)
- **RAM**: 32GB system memory
- **Python**: 3.10
- **CUDA**: 12.4
- **Storage**: SSD with 20GB+ free space

### Performance Expectations

| Sequence Length | Resolution | Inference Time | Peak VRAM | Hardware |
|-----------------|------------|----------------|-----------|----------|
| 8 frames        | 518×518    | ~5-7 sec       | ~7GB      | RTX 3060 |
| 17 frames       | 518×518    | ~10-12 sec     | ~10GB     | RTX 3060 |
| 17 frames       | 644×644    | ~14-17 sec     | ~12GB     | RTX 3060 |
| 32 frames       | 518×518    | ~20-25 sec     | ~14GB     | RTX 4080 |
| 64 frames       | 518×518    | ~40-50 sec     | ~16GB     | RTX 4080 |

*Times measured with fp16 precision and batch_size=16*

---

## 🔧 Troubleshooting

### Common Issues

**Problem: Out of memory (OOM) error**
- **Solution**:
  - Reduce `batch_size` in HWMInference (try 8 or 4)
  - Use smaller images with PreprocessImagesForHWM
  - Use `fp16` precision instead of `fp32`
  - Process fewer frames at once

**Problem: 3D viewers show black screen**
- **Solution**:
  - Hard refresh browser (Ctrl+Shift+R)
  - Check browser console (F12) for errors
  - Ensure PLY files exist at specified paths
  - Try closing and reopening ComfyUI

**Problem: Point cloud is upside down**
- **Solution**: This is fixed in latest version. Pull latest code.

**Problem: Colors don't match point cloud**
- **Solution**: Automatic resizing handles this. Ensure using latest version.

**Problem: Viewers are offset from nodes**
- **Solution**: Fixed in latest version. Hard refresh browser.

---

## 🎓 Best Practices

### For Best 3D Reconstruction:

1. **Use 12-24 frames** for optimal quality/speed balance
2. **Preprocess images** to 518×518 or 644×644
3. **Ensure good camera motion** - forward + orbital movement
4. **Maintain 50-70% overlap** between consecutive frames
5. **Use confidence filtering** (85-95 percentile) to remove noise
6. **Start with lower resolution** for testing, then increase

### For Memory Efficiency:

1. **Use fp16 precision** (50% memory reduction)
2. **Adjust batch_size** based on VRAM (8-16 for 12GB)
3. **Preprocess to 518×518** instead of higher resolutions
4. **Process in smaller sequences** if dealing with 100+ frames

### For Export Quality:

1. **Save point clouds with colors AND normals** for best results
2. **Use PLY format** for point clouds (supports all features)
3. **Enable scale filtering** for Gaussian splats (removes outliers)
4. **Use EXR for depth** if importing to 3D software
5. **Export camera params** for further reconstruction/optimization

---

## 📚 Example Workflows

### Basic 3D Reconstruction
```
LoadImage → PreprocessImagesForHWM → HWMInference → SavePointCloud
                                       ↓
                                    Preview3DGS
```

### Complete Pipeline with Visualization
```
LoadImage → PreprocessImagesForHWM → HWMInference →
                                       ├─ depth → VisualizeDepth → PreviewImage
                                       ├─ normals → VisualizeNormals → PreviewImage
                                       ├─ points3d → SavePointCloud → PreviewPointCloud
                                       ├─ gaussians → Save3DGaussians → Preview3DGS
                                       └─ poses + intrinsics → SaveCameraParams
```

---

## 🗺️ Roadmap

### ✅ Phase 1: Core Functionality (Complete)
- ✅ 11 essential nodes
- ✅ Single-pass inference with batch processing
- ✅ Multiple export formats
- ✅ Interactive 3D viewers
- ✅ Confidence filtering
- ✅ Automatic preprocessing
- ✅ Windows/Linux installation

### 🚧 Phase 2: Advanced Features (In Progress)
- [ ] Prior input nodes (poses, depth, intrinsics)
- [ ] Gaussian optimization/refinement
- [ ] Video-to-3D conversion (automatic frame extraction)
- [ ] Advanced camera controls for viewers

### 🔮 Phase 3: Ecosystem Integration (Planned)
- [ ] Mesh reconstruction from point clouds
- [ ] ControlNet depth conditioning
- [ ] Real-time preview during inference
- [ ] Multi-sequence batch processing

---

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The HunyuanWorld-Mirror model is developed by Tencent and licensed separately.

---

## 🙏 Acknowledgments

- [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) for the HunyuanWorld-Mirror model
- [gsplat](https://github.com/nerfstudio-project/gsplat) for 3D Gaussian Splatting implementation
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the amazing node-based interface
- [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) for 3D viewer reference implementation

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror/discussions)

---

## 📝 Citation

If you use this node pack in your research or projects, please cite:

```bibtex
@software{comfyui_hunyuanworld_mirror,
  title = {ComfyUI-HunyuanWorld-Mirror Node Pack},
  author = {Cedar Connor},
  year = {2025},
  url = {https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror}
}
```

---

**Made with ❤️ for the ComfyUI community**
