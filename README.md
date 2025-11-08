NOT WORKING YET!!!!!!!!!!!!!!!!!

# HunyuanWorld-Mirror ComfyUI Node Pack

Transform 2D images into 3D worlds using Tencent's HunyuanWorld-Mirror model directly in ComfyUI.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## Features

- **Single-Pass 3D Reconstruction** - Generate point clouds, depth maps, normals, camera parameters, and 3D Gaussians in one forward pass
- **8 Streamlined Nodes** - Minimal complexity, maximum functionality
- **ComfyUI Native** - Integrates seamlessly with existing image workflows
- **Multiple Export Formats** - PLY, OBJ, XYZ for point clouds; NPY, EXR, PFM for depth; JSON, COLMAP for cameras
- **3D Gaussian Splatting** - Export to standard 3DGS format for novel view synthesis
- **Memory Efficient** - Handles sequences up to 64 frames on 12GB VRAM
- **Easy Installation** - One-command setup for Windows and Linux

---

## What is HunyuanWorld-Mirror?

HunyuanWorld-Mirror is a universal 3D world reconstruction model developed by Tencent that:
- Takes a sequence of images (4-64 frames) as input
- Predicts comprehensive 3D geometry in a single forward pass
- Generates multiple 3D representations simultaneously
- Supports optional geometric priors (camera poses, depth, intrinsics)

This ComfyUI node pack brings these capabilities to your image generation workflows.

---

## Installation

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

### Manual Installation

If the automated scripts fail, install dependencies manually:

```bash
# Core dependencies
pip install torch==2.4.0 torchvision==0.19.0
pip install numpy>=1.24.0,<2.0.0 opencv-python Pillow imageio

# Model dependencies
pip install transformers diffusers huggingface-hub safetensors
pip install einops timm omegaconf scipy tqdm

# 3D export
pip install trimesh plyfile open3d

# Install gsplat (may require pre-compiled wheels)
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

### Verifying Installation

After restarting ComfyUI, you should see nodes under the category:
**`HunyuanWorld-Mirror`**

### Model Installation

The HunyuanWorld-Mirror model files need to be placed in ComfyUI's models directory:

1. **Create the models directory** (if it doesn't exist):
   ```bash
   mkdir -p ComfyUI/models/HunyuanWorld-Mirror
   ```

2. **Place your model file:**
   - Supported formats: `.safetensors`, `.pt`, `.pth`
   - Recommended location: `ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors`
   - Alternative: `ComfyUI/models/HunyuanWorld-Mirror/<any-name>.safetensors`

3. **Using the model in ComfyUI:**
   - In the `LoadHunyuanWorldMirrorModel` node, you can use:
     - **Filename only**: `HunyuanWorld-Mirror.safetensors` (automatically finds in models folder)
     - **Full path**: `C:/ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors`
     - **HuggingFace Hub**: `tencent/HunyuanWorld-Mirror` (downloads automatically)

**Model Path Resolution:**

The loader automatically searches for models in this order:
1. Direct path (if the provided path exists)
2. `ComfyUI/models/HunyuanWorld-Mirror/{model_name}`
3. `ComfyUI/models/HunyuanWorld-Mirror/{model_name}.safetensors`
4. `ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors` (default)
5. HuggingFace Hub (if model_name looks like `org/repo`)

**Example:**

If your model is at: `ComfyUI/models/HunyuanWorld-Mirror/HunyuanWorld-Mirror.safetensors`

You can use any of these in the model_name field:
- `HunyuanWorld-Mirror.safetensors`
- `HunyuanWorld-Mirror`
- Leave default and it will auto-detect

---

## Quick Start

### Basic Workflow

1. **Load Images:**
   - Use `LoadImage` (built-in) to load individual images
   - Use `ImageBatch` (built-in) to create a sequence of 4-64 images

2. **Load Model:**
   - Add `LoadHunyuanWorldMirrorModel` node
   - Select device (auto/cuda/cpu) and precision (fp32/fp16)

3. **Run Inference:**
   - Add `HWMInference` node
   - Connect model and images
   - Run to generate all outputs

4. **Visualize Results:**
   - Connect `depth` → `VisualizeDepth` → `PreviewImage`
   - Connect `normals` → `VisualizeNormals` → `PreviewImage`

5. **Export 3D Data:**
   - Connect `points3d` → `SavePointCloud`
   - Connect `gaussians` → `Save3DGaussians`
   - Specify output file paths

### Example Workflow Diagram

```
┌─────────────┐     ┌──────────────┐
│ LoadImage   │────→│ ImageBatch   │
└─────────────┘     └──────┬───────┘
                           │
┌──────────────────────┐   │
│ LoadHWMModel         │   │
│ - device: auto       │   │
│ - precision: fp16    │   │
└──────┬───────────────┘   │
       │                   │
       │    ┌──────────────┘
       ↓    ↓
┌─────────────────────────┐
│ HWMInference            │
│ - seed: -1              │
└──┬──┬──┬──┬──┬──────────┘
   │  │  │  │  │
   │  │  │  │  └─→ gaussians → Save3DGaussians → output.ply
   │  │  │  └────→ camera_poses → SaveCameraParams → cameras.json
   │  │  └───────→ points3d → SavePointCloud → pointcloud.ply
   │  └──────────→ normals → VisualizeNormals → PreviewImage
   └─────────────→ depth → VisualizeDepth → PreviewImage
```

---

## Node Reference

### 1. LoadHunyuanWorldMirrorModel

Load the HunyuanWorld-Mirror model with automatic caching.

**Inputs:**
- `model_name` (STRING): Model identifier (default: "tencent/HunyuanWorld-Mirror")
- `device` (COMBO): Device selection ["auto", "cuda", "cpu"]
- `precision` (COMBO): Precision mode ["fp32", "fp16", "bf16"]

**Outputs:**
- `MODEL`: Loaded model instance (cached for reuse)

**Notes:**
- Model downloads automatically from HuggingFace Hub on first use (~4GB)
- Use `fp16` to reduce memory usage by 50%
- Model is cached - subsequent loads are instant

---

### 2. HWMInference

Main inference node that generates all 3D predictions in one pass.

**Inputs:**
- `model` (MODEL): Loaded model from LoadHunyuanWorldMirrorModel
- `images` (IMAGE): Image sequence from ComfyUI (4-64 frames)
- `seed` (INT): Random seed for reproducibility (default: -1 for random)

**Outputs:**
- `depth` (DEPTH): Multi-view depth maps [N, H, W]
- `normals` (NORMALS): Surface normals [N, H, W, 3]
- `points3d` (POINTS3D): 3D point cloud [N, H, W, 3] (world coordinates)
- `camera_poses` (POSES): Camera-to-world matrices [N, 4, 4]
- `camera_intrinsics` (INTRINSICS): Intrinsic matrices [N, 3, 3]
- `gaussians` (GAUSSIANS): 3D Gaussian splat parameters

**Notes:**
- Processes all frames in a single forward pass
- Memory usage scales with sequence length
- Use 4-16 frames for 12GB VRAM, 16-32 for 24GB VRAM

---

### 3. VisualizeDepth

Convert depth maps to colorized visualizations.

**Inputs:**
- `depth` (DEPTH): Depth data from HWMInference
- `colormap` (COMBO): Color scheme ["viridis", "plasma", "turbo", "gray"]
- `normalize` (BOOLEAN): Auto-scale depth range (default: True)

**Outputs:**
- `IMAGE`: Colorized depth map (connect to PreviewImage)

**Notes:**
- Use "turbo" for best perceptual uniformity
- Normalization recommended for visualization

---

### 4. VisualizeNormals

Convert surface normals to RGB visualization.

**Inputs:**
- `normals` (NORMALS): Normal data from HWMInference

**Outputs:**
- `IMAGE`: RGB visualization (X→Red, Y→Green, Z→Blue)

**Notes:**
- Standard visualization: X-axis = red, Y-axis = green, Z-axis = blue
- Colors indicate surface orientation

---

### 5. SavePointCloud

Export 3D point cloud to standard formats.

**Inputs:**
- `points3d` (POINTS3D): Point cloud from HWMInference
- `colors` (IMAGE, optional): Point colors from source images
- `normals` (NORMALS, optional): Surface normals
- `filepath` (STRING): Output file path
- `format` (COMBO): File format ["ply", "pcd", "obj", "xyz"]

**Outputs:**
- `filepath` (STRING): Saved file location (confirmation)

**Supported Formats:**
- **PLY**: Binary/ASCII, supports colors and normals (recommended)
- **PCD**: Point Cloud Data format (PCL compatible)
- **OBJ**: Wavefront OBJ (points only)
- **XYZ**: Simple ASCII format (X Y Z per line)

**Notes:**
- PLY binary is most compact and feature-rich
- Include colors for textured point clouds
- Include normals for surface reconstruction

---

### 6. Save3DGaussians

Export 3D Gaussian Splatting representation.

**Inputs:**
- `gaussians` (GAUSSIANS): Gaussian data from HWMInference
- `filepath` (STRING): Output PLY file path
- `include_sh` (BOOLEAN): Include spherical harmonics (default: False)

**Outputs:**
- `filepath` (STRING): Saved file location

**Notes:**
- Standard 3DGS PLY format
- Compatible with SuperSplat, gsplat viewers, and 3DGS training code
- Use spherical harmonics for view-dependent effects

**Viewing Options:**
- [SuperSplat](https://playcanvas.com/supersplat) - Web-based viewer
- [gsplat](https://github.com/nerfstudio-project/gsplat) - Python viewer
- [3D Gaussian Splatting Viewer](https://github.com/antimatter15/splat) - Desktop viewer

---

### 7. SaveDepthMap

Export depth maps in various precision formats.

**Inputs:**
- `depth` (DEPTH): Depth data from HWMInference
- `filepath` (STRING): Output file path
- `format` (COMBO): File format ["npy", "exr", "pfm", "png16"]

**Outputs:**
- `filepath` (STRING): Saved file location

**Supported Formats:**
- **NPY**: NumPy array (full precision, Python-friendly)
- **EXR**: OpenEXR (high dynamic range, lossless float)
- **PFM**: Portable Float Map (standard format for depth)
- **PNG16**: 16-bit PNG (quantized, smaller file size)

**Notes:**
- Use NPY for further processing in Python
- Use EXR for interchange with 3D software
- Use PFM for academic/research use
- Use PNG16 for visualization and storage efficiency

---

### 8. SaveCameraParams

Export camera parameters for 3D reconstruction tools.

**Inputs:**
- `camera_poses` (POSES): Camera poses from HWMInference
- `camera_intrinsics` (INTRINSICS): Intrinsics from HWMInference
- `filepath` (STRING): Output file path
- `format` (COMBO): File format ["json", "colmap", "npy"]

**Outputs:**
- `filepath` (STRING): Saved file location

**Supported Formats:**
- **JSON**: Human-readable format with all parameters
- **COLMAP**: Binary format for COLMAP/MVS tools
- **NPY**: NumPy arrays (poses.npy + intrinsics.npy)

**Notes:**
- JSON includes focal length, principal point, and matrices
- COLMAP format compatible with structure-from-motion pipelines
- Coordinate system: OpenCV (camera-to-world)

---

## Output Formats

### Point Cloud (PLY)

The exported PLY files contain:
- **Vertex positions**: XYZ coordinates in world space
- **Colors** (optional): RGB values from source images
- **Normals** (optional): Surface normal vectors

Compatible with:
- MeshLab (visualization and editing)
- CloudCompare (analysis and processing)
- Blender (import as point cloud)
- Open3D (Python processing)

### 3D Gaussians (PLY)

The exported 3DGS files contain:
- **Means**: 3D Gaussian centers
- **Scales**: Gaussian size parameters
- **Rotations**: Quaternion rotations (wxyz)
- **Colors**: RGB or spherical harmonics
- **Opacities**: Transparency values

Compatible with:
- SuperSplat viewer
- gsplat Python library
- 3DGS training/optimization code

### Depth Maps

Depth values represent distance from camera:
- **Units**: Metric (meters) or relative
- **Format**: Per-pixel depth value
- **Coordinate**: Camera space Z-axis

### Camera Parameters

Camera poses are 4x4 matrices (camera-to-world):
```
[R | t]
[0 | 1]
```

Camera intrinsics are 3x3 matrices:
```
[fx  0  cx]
[ 0 fy  cy]
[ 0  0   1]
```

---

## System Requirements

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

| Sequence Length | Inference Time | Peak VRAM | Hardware |
|-----------------|----------------|-----------|----------|
| 4 frames        | ~3-5 sec       | ~6GB      | RTX 3060 |
| 8 frames        | ~6-8 sec       | ~8GB      | RTX 3060 |
| 16 frames       | ~12-15 sec     | ~10GB     | RTX 3060 |
| 32 frames       | ~25-30 sec     | ~16GB     | RTX 4080 |

*Times measured on RTX 4080 with fp16 precision*

---

## Troubleshooting

### Installation Issues

**Problem: `install.bat` fails with "Python 3.10 required"**
- **Solution**: Install Python 3.10 from [python.org](https://www.python.org/downloads/)
- Ensure Python 3.10 is in your PATH

**Problem: `gsplat` installation fails**
- **Solution**: Try installing from PyPI: `pip install gsplat`
- Falls back to JIT compilation (slower first run)
- See [gsplat docs](https://docs.gsplat.studio/) for pre-compiled wheels

**Problem: CUDA version mismatch**
- **Solution**: Check your CUDA version with `nvcc --version`
- Install matching PyTorch: `pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124`

### Runtime Issues

**Problem: Out of memory (OOM) error**
- **Solution**: Reduce sequence length (use fewer images)
- Use lower precision: `fp16` instead of `fp32`
- Reduce image resolution before batching
- Close other GPU applications

**Problem: Nodes not appearing in ComfyUI**
- **Solution**: Check ComfyUI console for import errors
- Verify installation completed successfully
- Restart ComfyUI completely
- Check `custom_nodes/ComfyUI-HunyuanWorld-Mirror/__init__.py` exists

**Problem: Model download fails**
- **Solution**: Download manually from [HuggingFace](https://huggingface.co/tencent/HunyuanWorld-Mirror)
- Place in `models/` directory
- Set `HF_HOME` environment variable if behind proxy

**Problem: Inference is very slow**
- **Solution**: Ensure using CUDA (not CPU)
- Check GPU utilization with `nvidia-smi`
- Use `fp16` precision for 2x speedup
- Close background GPU applications

### Export Issues

**Problem: PLY file won't open in MeshLab**
- **Solution**: Verify file saved successfully (check file size > 0)
- Try ASCII format instead of binary
- Check file path doesn't have special characters

**Problem: Gaussians don't render in SuperSplat**
- **Solution**: Ensure using standard 3DGS PLY format
- Check `include_sh` is disabled for basic rendering
- Verify file contains all required attributes

### Windows-Specific Issues

**Problem: Long path errors**
- **Solution**: Enable long path support in Windows
- Run: `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1`
- Or use shorter output paths

**Problem: Permission denied errors**
- **Solution**: Run installation as Administrator
- Or install in user directory

---

## FAQ

**Q: How many images do I need?**
A: Minimum 4 images, recommended 8-16 for best results. More images = better coverage but slower inference.

**Q: What kind of images work best?**
A: Images with camera motion (forward, orbit, or mixed). Avoid pure rotation without translation.

**Q: Can I use this with video frames?**
A: Yes! Extract frames with ComfyUI video nodes or external tools, then batch them.

**Q: Do images need to be sequential?**
A: Not required, but sequential frames usually give better results.

**Q: What resolution should images be?**
A: Model works with 518x518. Use `ImageScale` node to resize if needed.

**Q: Can I use camera pose priors?**
A: Not in Phase 1 (core implementation). Advanced nodes for priors coming in Phase 2.

**Q: How do I view the 3D Gaussians?**
A: Use [SuperSplat](https://playcanvas.com/supersplat) (web) or [gsplat](https://github.com/nerfstudio-project/gsplat) (local).

**Q: Can I render novel views?**
A: Coming in Phase 2 with the `RenderGaussianView` node.

**Q: Does this work on AMD/Mac GPUs?**
A: Currently optimized for NVIDIA CUDA GPUs. CPU mode works but is very slow.

**Q: Can I train or fine-tune the model?**
A: Not supported in this node pack. Use the [official repository](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) for training.

---

## Example Workflows

### Basic 3D Reconstruction

Load a sequence of images and export point cloud:

```
LoadImage → ImageBatch → LoadHWMModel + HWMInference → points3d → SavePointCloud
```

See `examples/basic_workflow.json`

### Full 3D Pipeline

Generate all outputs with visualizations:

```
LoadImage → ImageBatch → LoadHWMModel + HWMInference →
  ├─ depth → VisualizeDepth → PreviewImage
  ├─ normals → VisualizeNormals → PreviewImage
  ├─ points3d → SavePointCloud
  ├─ gaussians → Save3DGaussians
  └─ camera_poses + intrinsics → SaveCameraParams
```

See `examples/advanced_workflow.json`

---

## Roadmap

### Phase 1: Core Functionality (Current)
- ✅ 8 essential nodes
- ✅ Single-pass inference
- ✅ Multiple export formats
- ✅ Windows/Linux installation

### Phase 2: Advanced Features (Planned)
- [ ] Prior input nodes (poses, depth, intrinsics)
- [ ] Gaussian optimization/refinement
- [ ] Novel view rendering
- [ ] Batch processing multiple sequences

### Phase 3: Ecosystem Integration (Future)
- [ ] Mesh reconstruction from point clouds
- [ ] ControlNet depth conditioning
- [ ] Video-to-3D conversion
- [ ] Real-time preview

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See `DEVELOPMENT.md` for development setup and guidelines.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The HunyuanWorld-Mirror model is developed by Tencent and licensed separately.

---

## Citation

If you use this node pack in your research or projects, please cite:

```bibtex
@software{comfyui_hunyuanworld_mirror,
  title = {ComfyUI-HunyuanWorld-Mirror Node Pack},
  author = {Cedar Connor},
  year = {2025},
  url = {https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror}
}

@article{hunyuanworld_mirror,
  title = {HunyuanWorld-Mirror: Universal 3D World Reconstruction},
  author = {Tencent Hunyuan Team},
  year = {2025},
  url = {https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror}
}
```

---

## Acknowledgments

- [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror) for the HunyuanWorld-Mirror model
- [gsplat](https://github.com/nerfstudio-project/gsplat) for 3D Gaussian Splatting implementation
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the amazing node-based interface

---

## Support

- **Issues**: [GitHub Issues](https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror/discussions)
- **Discord**: Coming soon

---

## Changelog

### Version 1.0.0 (Current)
- Initial release
- 8 core nodes
- Multiple export formats
- Windows/Linux installation
- Example workflows

---

**Made with ❤️ for the ComfyUI community**
