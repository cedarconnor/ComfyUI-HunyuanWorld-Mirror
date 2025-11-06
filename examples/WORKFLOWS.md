# HunyuanWorld-Mirror Example Workflows

This document describes how to set up workflows using the HunyuanWorld-Mirror nodes in ComfyUI.

## Workflow 1: Basic 3D Reconstruction

**Goal:** Convert a sequence of images into a 3D point cloud

### Nodes Required:
1. `LoadImage` (ComfyUI built-in)
2. `ImageBatch` (ComfyUI built-in) - if loading multiple images
3. `LoadHunyuanWorldMirrorModel` (HunyuanWorld-Mirror)
4. `HWMInference` (HunyuanWorld-Mirror)
5. `SavePointCloud` (HunyuanWorld-Mirror)

### Setup:

```
1. LoadImage
   └─> ImageBatch (if multiple images)
       └─> HWMInference ──┐
                          ├─> points3d ──> SavePointCloud
   LoadHWMModel ──────────┘

```

### Configuration:

**LoadHWMModel:**
- model_name: `tencent/HunyuanWorld-Mirror`
- device: `auto`
- precision: `fp16` (recommended for memory efficiency)

**HWMInference:**
- images: Connect from ImageBatch
- model: Connect from LoadHWMModel
- seed: `-1` (random) or any number for reproducibility

**SavePointCloud:**
- points3d: Connect from HWMInference → points3d
- filepath: `./output/pointcloud.ply`
- format: `ply`
- colors: (optional) Connect original images for colored points
- normals: (optional) Connect from HWMInference → normals

---

## Workflow 2: Full 3D Pipeline with Visualizations

**Goal:** Generate all outputs (depth, normals, point cloud, Gaussians) with previews

### Nodes Required:
1. `LoadImage` (built-in)
2. `ImageBatch` (built-in)
3. `LoadHunyuanWorldMirrorModel`
4. `HWMInference`
5. `VisualizeDepth`
6. `VisualizeNormals`
7. `PreviewImage` (built-in) × 2
8. `SavePointCloud`
9. `Save3DGaussians`
10. `SaveCameraParams`

### Setup:

```
LoadImage → ImageBatch ──┐
                         │
LoadHWMModel ────────────┼──> HWMInference ──┬──> depth ──> VisualizeDepth ──> PreviewImage
                         │                    │
                         │                    ├──> normals ──> VisualizeNormals ──> PreviewImage
                         │                    │
                         │                    ├──> points3d ──> SavePointCloud
                         │                    │
                         │                    ├──> gaussians ──> Save3DGaussians
                         │                    │
                         │                    └──> camera_poses ──┐
                         │                         camera_intrinsics ──> SaveCameraParams
                         │
```

### Configuration:

**VisualizeDepth:**
- depth: Connect from HWMInference → depth
- colormap: `turbo` (good perceptual uniformity)
- normalize: `True`

**VisualizeNormals:**
- normals: Connect from HWMInference → normals

**SavePointCloud:**
- filepath: `./output/pointcloud.ply`
- format: `ply`

**Save3DGaussians:**
- filepath: `./output/gaussians.ply`
- include_sh: `False` (basic mode)

**SaveCameraParams:**
- filepath: `./output/cameras.json`
- format: `json`

---

## Workflow 3: Depth Map Generation

**Goal:** Generate and save depth maps only

### Nodes Required:
1. `LoadImage`
2. `ImageBatch`
3. `LoadHunyuanWorldMirrorModel`
4. `HWMInference`
5. `VisualizeDepth`
6. `SaveDepthMap`
7. `PreviewImage`

### Setup:

```
LoadImage → ImageBatch ──> HWMInference ──> depth ──┬──> VisualizeDepth ──> PreviewImage
                              ↑                      │
LoadHWMModel ─────────────────┘                      └──> SaveDepthMap
```

### Configuration:

**SaveDepthMap:**
- depth: Connect from HWMInference → depth
- filepath: `./output/depth.npy`
- format: `npy` (full precision) or `png16` (compact)

---

## Tips for Best Results

### Image Preparation

1. **Number of Images:**
   - Minimum: 4 images
   - Recommended: 8-16 images
   - Maximum: 64 images (limited by memory)

2. **Image Quality:**
   - Use clear, well-lit images
   - Avoid motion blur
   - Consistent lighting across sequence

3. **Camera Motion:**
   - Forward motion works best
   - Orbital/circular motion also good
   - Avoid pure rotation without translation

4. **Resolution:**
   - Model works with 518×518 internally
   - Use `ImageScale` node to resize if needed
   - Higher input resolution = better quality (but slower)

### Memory Management

**For 12GB VRAM (RTX 3060):**
- Precision: `fp16`
- Max frames: 16 at 512×512
- Close other GPU applications

**For 24GB VRAM (RTX 4090):**
- Precision: `fp32` or `fp16`
- Max frames: 32-64 at 512×512

**Out of Memory?**
- Reduce number of images
- Use `fp16` precision
- Lower image resolution
- Restart ComfyUI to clear cache

### Output Usage

**Point Clouds (.ply):**
- Open in: MeshLab, CloudCompare, Blender
- Good for: Inspection, measurement, further processing

**3D Gaussians (.ply):**
- Open in: [SuperSplat](https://playcanvas.com/supersplat) (web)
- Good for: Novel view synthesis, real-time rendering

**Depth Maps:**
- `.npy`: Python/NumPy processing
- `.exr`: 3D software (Blender, Maya)
- `.png16`: Lightweight storage

**Camera Parameters:**
- `.json`: Human-readable, easy to parse
- `.npy`: Direct NumPy loading
- Use for: Camera path animation, novel view rendering

---

## Troubleshooting

### Issue: Nodes not appearing
**Solution:**
1. Check ComfyUI console for import errors
2. Verify installation: `pip list | grep hunyuan`
3. Restart ComfyUI completely

### Issue: Inference very slow
**Solution:**
1. Check device is set to `auto` or `cuda`
2. Use `fp16` precision
3. Monitor GPU usage: `nvidia-smi`

### Issue: Output files empty or corrupted
**Solution:**
1. Check output directory exists
2. Verify file path doesn't have special characters
3. Try different export format

### Issue: Low quality results
**Solution:**
1. Use more images (12-16 recommended)
2. Ensure good camera motion
3. Check input image quality
4. Try higher resolution inputs

---

## Advanced Usage

### Custom Output Paths

Use date/time in filenames:
```
filepath: ./output/pointcloud_2025-01-15_14-30.ply
```

### Batch Processing

To process multiple sequences:
1. Create separate workflows
2. Use different output paths
3. Queue all workflows in ComfyUI

### Integration with Other Nodes

**Depth for ControlNet:**
- VisualizeDepth → ControlNet Preprocessor
- Use for conditioning Stable Diffusion

**Point Clouds in Blender:**
1. Save as PLY with colors
2. Import in Blender (File → Import → PLY)
3. Use for reference geometry

---

## Example Use Cases

1. **3D Asset Creation:**
   - Capture object from multiple angles
   - Generate point cloud
   - Convert to mesh in Blender

2. **Environment Reconstruction:**
   - Record video walkthrough
   - Extract frames
   - Generate 3D scene

3. **Novel View Synthesis:**
   - Generate Gaussian splats
   - Render from arbitrary camera positions
   - Create smooth camera paths

4. **Depth Estimation:**
   - Process video frames
   - Extract depth maps
   - Use for visual effects

---

## Resources

- **HunyuanWorld-Mirror:** https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
- **SuperSplat Viewer:** https://playcanvas.com/supersplat
- **MeshLab:** https://www.meshlab.net/
- **CloudCompare:** https://www.cloudcompare.org/

---

**Need Help?**

Open an issue: https://github.com/cedarconnor/ComfyUI-HunyuanWorld-Mirror/issues
