# HunyuanWorld-Mirror ComfyUI Node Pack - Design Document

**Version:** 1.0  
**Date:** November 5, 2025  
**Target Platform:** Windows (with Linux compatibility)  
**Status:** Planning Phase

---

## Executive Summary

This document outlines the design and implementation plan for integrating Tencent's HunyuanWorld-Mirror - a universal 3D world reconstruction model - into ComfyUI as a custom node pack. The integration will enable ComfyUI users to perform comprehensive 3D geometric prediction including point clouds, depth maps, surface normals, camera parameters, and 3D Gaussian Splatting from image sequences.

**Key Goals:**
- Provide intuitive ComfyUI nodes for 3D reconstruction
- Support Windows installation with pre-compiled binaries
- Enable flexible prior conditioning (poses, depth, intrinsics)
- Output multiple 3D representations in standard formats
- Maintain performance and memory efficiency

---

## 1. Project Overview

### 1.1 Background

**HunyuanWorld-Mirror** is a versatile feed-forward model that:
- Processes sequences of images (video frames or image collections)
- Accepts optional geometric priors (camera poses, depth maps, intrinsics)
- Generates multiple 3D representations in a single forward pass
- Produces high-quality 3D reconstructions for novel view synthesis

**Model Capabilities:**
- 3D point cloud generation (world coordinates)
- Multi-view depth estimation
- Surface normal prediction
- Camera pose estimation (camera-to-world)
- Camera intrinsics estimation
- 3D Gaussian Splatting representation

### 1.2 Target Users

- **3D Artists:** Creating 3D assets from 2D images/videos
- **Game Developers:** Generating 3D environments from reference images
- **Researchers:** Experimenting with 3D reconstruction workflows
- **ComfyUI Power Users:** Building complex multi-modal generation pipelines

### 1.3 Success Criteria

1. **Ease of Installation:** One-click/script installation on Windows
2. **User-Friendly Interface:** Intuitive node connections matching ComfyUI conventions
3. **Performance:** Efficient GPU memory usage (target: 12GB VRAM for standard scenes)
4. **Compatibility:** Works with existing ComfyUI image processing nodes
5. **Output Quality:** Preserves model quality with proper format conversions
6. **Documentation:** Clear examples and tutorials for common use cases

---

## 2. Technical Architecture

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ComfyUI Core                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          HunyuanWorld-Mirror Node Pack                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Input Layer                                          │  │
│  │  - LoadHWMModel                                       │  │
│  │  - PrepareImageSequence                               │  │
│  │  - LoadCameraPoses (optional)                         │  │
│  │  - LoadDepthMaps (optional)                           │  │
│  │  - LoadIntrinsics (optional)                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Processing Layer                                     │  │
│  │  - HWMInference (main prediction)                     │  │
│  │  - TensorConverter (format handling)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                 │
│                            ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Output Layer                                         │  │
│  │  - PreviewPointCloud                                  │  │
│  │  - SavePointCloud (.ply)                              │  │
│  │  - PreviewDepth                                       │  │
│  │  - SaveDepth (.npy, .exr)                             │  │
│  │  - PreviewNormals                                     │  │
│  │  - SaveNormals                                        │  │
│  │  - SaveCameraParameters (.json, COLMAP)               │  │
│  │  - Save3DGaussians (.ply)                             │  │
│  │  - PreviewGaussians (render view)                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  External Libraries                                         │
│  - PyTorch 2.4.0 (CUDA 12.4)                               │
│  - gsplat (3D Gaussian Splatting)                          │
│  - HuggingFace Hub (model loading)                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Input Images (ComfyUI)           Optional Priors
[B, H, W, C] (0-1 float)        [poses, depth, intrinsics]
        │                                │
        ▼                                ▼
 Format Conversion              Prior Preprocessing
[1, N, 3, H, W]                 [1, N, ...] format
        │                                │
        └────────────┬───────────────────┘
                     ▼
            HWM Model Inference
            condition=[p, d, i]
                     │
                     ▼
        ┌────────────┴────────────┐
        ▼            ▼             ▼
    Geometry    Camera Params   Gaussians
    Outputs        Outputs       Outputs
        │            │             │
        ▼            ▼             ▼
  [pts3d,      [poses,          [means,
   depth,       intrs]           scales,
   normals]                      quats,
                                 colors,
                                 opacities]
        │            │             │
        └────────────┴─────────────┘
                     ▼
         Post-processing & Export
         (PLY, NPY, JSON, EXR, COLMAP)
```

### 2.3 Node Type System

Custom data types for node connections:

```python
# Custom Types
HWMIRROR_MODEL     # The loaded model instance
IMAGE_SEQUENCE     # Batch of images [N, H, W, C]
POINTS3D           # 3D point cloud [N, H, W, 3]
DEPTH_MAP          # Depth map [N, H, W, 1]
NORMAL_MAP         # Normal map [N, H, W, 3]
CAMERA_POSES       # Camera poses [N, 4, 4]
CAMERA_INTRINSICS  # Intrinsics [N, 3, 3]
GAUSSIAN_SPLATS    # Dict of Gaussian parameters
```

---

## 3. Detailed Node Specifications

### 3.1 Input/Loader Nodes

#### 3.1.1 LoadHunyuanWorldMirror

**Purpose:** Load and cache the HunyuanWorld-Mirror model

**Inputs:**
- `model_name` (STRING): Model identifier (default: "tencent/HunyuanWorld-Mirror")
- `device` (COMBO): ["auto", "cuda", "cpu"]
- `precision` (COMBO): ["fp32", "fp16", "bf16"]
- `cache_dir` (STRING): Custom cache directory (optional)

**Outputs:**
- `model` (HWMIRROR_MODEL): Loaded model instance

**Implementation Notes:**
- Use HuggingFace Hub for automatic downloads
- Implement model caching to avoid reloading
- Check VRAM availability before loading
- Support half-precision for memory efficiency

**Error Handling:**
- Validate CUDA availability
- Check minimum VRAM (recommend 12GB)
- Graceful fallback if download fails
- Clear error messages for missing dependencies

#### 3.1.2 PrepareImageSequence

**Purpose:** Convert ComfyUI images to HWM format and preprocess

**Inputs:**
- `images` (IMAGE): ComfyUI image batch
- `target_size` (INT): Resize images (default: 518)
- `maintain_aspect` (BOOLEAN): Keep aspect ratio (default: True)
- `fps` (FLOAT): Frame rate for video extraction (optional)

**Outputs:**
- `image_sequence` (IMAGE_SEQUENCE): Preprocessed images
- `original_size` (TUPLE): Original dimensions for post-processing

**Implementation Notes:**
- Convert from [B, H, W, C] to [1, N, 3, H, W]
- Normalize to [0, 1] range (if not already)
- Center crop or padding if maintaining aspect ratio
- Support variable sequence lengths

**Preprocessing Steps:**
1. Validate input format
2. Resize to target_size
3. RGB channel ordering verification
4. Normalize pixel values
5. Convert to torch tensor
6. Move to appropriate device

#### 3.1.3 LoadCameraPoses (Optional)

**Purpose:** Load pre-computed camera poses as priors

**Inputs:**
- `poses_path` (STRING): Path to .npy file
- `format` (COMBO): ["c2w", "w2c", "auto-detect"]
- `convention` (COMBO): ["opencv", "opengl"]

**Outputs:**
- `camera_poses` (CAMERA_POSES): Camera pose matrices [N, 4, 4]

**Implementation Notes:**
- Support multiple file formats (.npy, .json, .txt)
- Convert between coordinate conventions
- Validate pose matrix properties (rotation orthogonality)

#### 3.1.4 LoadDepthMaps (Optional)

**Purpose:** Load depth maps as priors

**Inputs:**
- `depth_path` (STRING): Path to depth data
- `depth_format` (COMBO): ["npy", "pfm", "exr", "png_16bit"]
- `normalize` (BOOLEAN): Normalize to [0, 1]

**Outputs:**
- `depth_maps` (DEPTH_MAP): Depth maps [N, H, W]

**Implementation Notes:**
- Support metric and relative depth
- Handle different depth scales
- Resize to match image sequence

#### 3.1.5 LoadIntrinsics (Optional)

**Purpose:** Load camera intrinsic parameters

**Inputs:**
- `intrinsics_path` (STRING): Path to intrinsics
- `format` (COMBO): ["matrix", "fov", "focal_length"]

**Outputs:**
- `intrinsics` (CAMERA_INTRINSICS): 3x3 intrinsic matrices [N, 3, 3]

**Implementation Notes:**
- Convert from various parameterizations
- Support per-frame or shared intrinsics
- Validate focal length and principal point

### 3.2 Processing Nodes

#### 3.2.1 HunyuanWorldMirrorInference

**Purpose:** Main inference node for 3D reconstruction

**Inputs:**
- `model` (HWMIRROR_MODEL): Loaded model
- `images` (IMAGE_SEQUENCE): Preprocessed images
- `use_pose_prior` (BOOLEAN): Enable pose conditioning
- `use_depth_prior` (BOOLEAN): Enable depth conditioning
- `use_intrinsic_prior` (BOOLEAN): Enable intrinsic conditioning
- `camera_poses` (CAMERA_POSES, optional): Pose priors
- `depth_maps` (DEPTH_MAP, optional): Depth priors
- `intrinsics` (CAMERA_INTRINSICS, optional): Intrinsic priors
- `seed` (INT): Random seed for reproducibility (default: -1)

**Outputs:**
- `points3d` (POINTS3D): 3D point cloud [N, H, W, 3]
- `depth` (DEPTH_MAP): Depth predictions [N, H, W, 1]
- `normals` (NORMAL_MAP): Surface normals [N, H, W, 3]
- `camera_poses` (CAMERA_POSES): Predicted/refined poses [N, 4, 4]
- `camera_intrinsics` (CAMERA_INTRINSICS): Predicted intrinsics [N, 3, 3]
- `gaussian_splats` (GAUSSIAN_SPLATS): 3D Gaussian parameters

**Implementation Notes:**
- Set condition flags based on prior availability
- Handle variable sequence lengths
- Clear GPU memory after inference
- Support batched processing for long sequences

**Performance Optimizations:**
- Use torch.no_grad() for inference
- Enable mixed precision if supported
- Implement chunked processing for memory constraints
- Cache model activations when possible

#### 3.2.2 RefineGaussians (Advanced)

**Purpose:** Optional Gaussian Splatting refinement

**Inputs:**
- `gaussian_splats` (GAUSSIAN_SPLATS): Initial Gaussians
- `images` (IMAGE_SEQUENCE): Reference images
- `camera_poses` (CAMERA_POSES): Camera parameters
- `iterations` (INT): Optimization iterations
- `learning_rate` (FLOAT): Optimization LR

**Outputs:**
- `refined_splats` (GAUSSIAN_SPLATS): Optimized Gaussians

**Implementation Notes:**
- Integrate gsplat optimization loop
- Support subset of frames for efficiency
- Monitor loss convergence

### 3.3 Output/Visualization Nodes

#### 3.3.1 PreviewPointCloud

**Purpose:** Interactive 3D point cloud visualization

**Inputs:**
- `points3d` (POINTS3D): Point cloud data
- `colors` (IMAGE_SEQUENCE, optional): Point colors from images
- `normals` (NORMAL_MAP, optional): Surface normals
- `view_angle` (COMBO): ["front", "top", "free"]
- `point_size` (FLOAT): Render point size

**Outputs:**
- `preview` (IMAGE): Rendered view of point cloud

**Implementation Notes:**
- Use matplotlib or plotly for rendering
- Support rotation and zoom
- Color by depth, normal, or image texture

#### 3.3.2 SavePointCloud

**Purpose:** Export point cloud to standard formats

**Inputs:**
- `points3d` (POINTS3D): Point cloud data
- `colors` (IMAGE_SEQUENCE, optional): Point colors
- `normals` (NORMAL_MAP, optional): Normal data
- `output_path` (STRING): Save location
- `format` (COMBO): ["ply", "pcd", "obj", "xyz"]
- `coordinate_system` (COMBO): ["world", "camera"]

**Outputs:**
- `file_path` (STRING): Saved file location

**Implementation Notes:**
- Support binary and ASCII PLY
- Include normal and color attributes
- Optionally filter outliers

#### 3.3.3 PreviewDepth

**Purpose:** Visualize depth maps with colormaps

**Inputs:**
- `depth` (DEPTH_MAP): Depth data
- `colormap` (COMBO): ["viridis", "plasma", "gray", "turbo"]
- `normalize` (BOOLEAN): Auto-scale depth range
- `min_depth` (FLOAT, optional): Manual min
- `max_depth` (FLOAT, optional): Manual max

**Outputs:**
- `depth_vis` (IMAGE): Colorized depth map

#### 3.3.4 SaveDepth

**Purpose:** Save depth maps in various formats

**Inputs:**
- `depth` (DEPTH_MAP): Depth data
- `output_path` (STRING): Save location
- `format` (COMBO): ["npy", "exr", "pfm", "png_16bit"]
- `scale` (FLOAT): Depth scaling factor

**Outputs:**
- `file_path` (STRING): Saved file location

**Implementation Notes:**
- NPY: Full precision numpy array
- EXR: High dynamic range format
- PFM: Portable float map
- PNG: 16-bit quantized depth

#### 3.3.5 PreviewNormals

**Purpose:** Visualize surface normals

**Inputs:**
- `normals` (NORMAL_MAP): Normal data
- `coordinate_space` (COMBO): ["camera", "world"]

**Outputs:**
- `normal_vis` (IMAGE): Colorized normal map (RGB = XYZ)

#### 3.3.6 SaveCameraParameters

**Purpose:** Export camera parameters

**Inputs:**
- `camera_poses` (CAMERA_POSES): Pose matrices
- `camera_intrinsics` (CAMERA_INTRINSICS): Intrinsic matrices
- `output_path` (STRING): Save location
- `format` (COMBO): ["json", "colmap", "numpy"]

**Outputs:**
- `file_path` (STRING): Saved file location

**Implementation Notes:**
- JSON: Human-readable format
- COLMAP: For structure-from-motion tools
- NumPy: Direct array serialization

#### 3.3.7 Save3DGaussians

**Purpose:** Export Gaussian Splatting representation

**Inputs:**
- `gaussian_splats` (GAUSSIAN_SPLATS): Gaussian parameters
- `output_path` (STRING): Save location (.ply)
- `include_sh` (BOOLEAN): Include spherical harmonics
- `max_sh_degree` (INT): SH degree (0-3)

**Outputs:**
- `file_path` (STRING): Saved PLY file

**Implementation Notes:**
- Standard 3DGS PLY format
- Compatible with viewers (SuperSplat, etc.)
- Include all Gaussian attributes

#### 3.3.8 RenderGaussianView

**Purpose:** Render novel views using Gaussians

**Inputs:**
- `gaussian_splats` (GAUSSIAN_SPLATS): Gaussian data
- `camera_pose` (CAMERA_POSES): Novel view pose
- `camera_intrinsics` (CAMERA_INTRINSICS): Camera parameters
- `width` (INT): Render width
- `height` (INT): Render height
- `background` (COMBO): ["white", "black", "transparent"]

**Outputs:**
- `rendered_image` (IMAGE): Novel view rendering

**Implementation Notes:**
- Use gsplat rasterizer
- Support anti-aliasing
- Hardware accelerated rendering

---

## 4. Installation Strategy

### 4.1 Windows Installation

**Prerequisites:**
- Python 3.10
- CUDA 12.4 Toolkit
- ComfyUI installed

**Installation Script (`install.bat`):**

```batch
@echo off
echo ============================================
echo HunyuanWorld-Mirror ComfyUI Node Pack
echo Installation Script for Windows
echo ============================================
echo.

REM Check Python version
python --version | findstr "3.10" >nul
if errorlevel 1 (
    echo ERROR: Python 3.10 is required
    echo Please install Python 3.10 and try again
    pause
    exit /b 1
)

REM Check CUDA
nvcc --version | findstr "12.4" >nul
if errorlevel 1 (
    echo WARNING: CUDA 12.4 not detected
    echo The installation may still work with CUDA 12.x
    echo.
)

echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Installing gsplat with pre-compiled wheels...
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
if errorlevel 1 (
    echo ERROR: Failed to install gsplat
    echo Try installing manually from PyPI: pip install gsplat
    pause
    exit /b 1
)

echo.
echo Downloading HunyuanWorld-Mirror model...
echo This may take several minutes (model is ~4GB)
echo.
python -m huggingface_hub.cli download tencent/HunyuanWorld-Mirror --local-dir ./models
if errorlevel 1 (
    echo WARNING: Model download failed
    echo The model will be downloaded automatically on first use
    echo.
)

echo.
echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo Restart ComfyUI to load the new nodes.
echo Nodes will appear under: HunyuanWorld-Mirror
echo.
pause
```

### 4.2 Linux Installation

**Installation Script (`install.sh`):**

```bash
#!/bin/bash

echo "============================================"
echo "HunyuanWorld-Mirror ComfyUI Node Pack"
echo "Installation Script for Linux"
echo "============================================"
echo ""

# Check Python version
if ! python --version | grep -q "3.10"; then
    echo "ERROR: Python 3.10 is required"
    exit 1
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA detected: $(nvcc --version | grep release)"
else
    echo "WARNING: CUDA not detected"
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt || {
    echo "ERROR: Failed to install dependencies"
    exit 1
}

echo ""
echo "Installing gsplat..."
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124 || {
    echo "ERROR: Failed to install gsplat"
    exit 1
}

echo ""
echo "Downloading HunyuanWorld-Mirror model..."
python -m huggingface_hub.cli download tencent/HunyuanWorld-Mirror --local-dir ./models || {
    echo "WARNING: Model download failed"
    echo "The model will be downloaded automatically on first use"
}

echo ""
echo "============================================"
echo "Installation Complete!"
echo "============================================"
echo ""
echo "Restart ComfyUI to load the new nodes."
echo "Nodes will appear under: HunyuanWorld-Mirror"
```

### 4.3 Dependencies

**requirements.txt:**

```
# Core dependencies
torch>=2.4.0
torchvision>=0.19.0
numpy>=1.24.0
opencv-python>=4.8.0

# Image/video processing
Pillow>=10.0.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.9

# Deep learning frameworks
einops>=0.7.0
timm>=0.9.0
transformers>=4.35.0
diffusers>=0.24.0

# Model loading and caching
huggingface-hub>=0.19.0
safetensors>=0.4.0

# 3D processing
trimesh>=4.0.0
plyfile>=1.0.0
open3d>=0.18.0

# Utilities
tqdm>=4.66.0
omegaconf>=2.3.0
scipy>=1.11.0

# For gsplat (installed separately)
ninja
jaxtyping
rich
```

### 4.4 Troubleshooting Guide

**Common Issues:**

1. **CUDA Version Mismatch**
   - Solution: Install PyTorch matching your CUDA version
   - Check: `nvcc --version` and `torch.version.cuda`

2. **Out of Memory**
   - Solution: Reduce image resolution or sequence length
   - Use half precision: `precision="fp16"`

3. **gsplat Installation Fails**
   - Solution: Try PyPI version: `pip install gsplat`
   - Falls back to JIT compilation

4. **Model Download Fails**
   - Solution: Manual download from HuggingFace
   - Set HF_TOKEN environment variable for gated models

5. **Node Not Appearing**
   - Solution: Check `__init__.py` node registration
   - Restart ComfyUI completely
   - Check console for import errors

---

## 5. Implementation Details

### 5.1 Tensor Format Conversions

**ComfyUI ↔ HunyuanWorld-Mirror Format Handling:**

```python
# utils/tensor_utils.py

import torch
import numpy as np
from typing import Tuple

def comfy_to_hwmirror(comfy_images: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI image format to HunyuanWorld-Mirror format.
    
    Args:
        comfy_images: [B, H, W, C] in range [0, 1]
    
    Returns:
        hwm_images: [1, N, 3, H, W] in range [0, 1]
    """
    # ComfyUI: [B, H, W, C] -> HWM: [1, N, 3, H, W]
    if comfy_images.dim() == 4:
        # [B, H, W, C] -> [B, C, H, W]
        hwm_images = comfy_images.permute(0, 3, 1, 2)
        # [B, C, H, W] -> [1, B, C, H, W]
        hwm_images = hwm_images.unsqueeze(0)
    else:
        raise ValueError(f"Expected 4D tensor, got {comfy_images.dim()}D")
    
    return hwm_images


def hwmirror_to_comfy(hwm_images: torch.Tensor) -> torch.Tensor:
    """
    Convert HunyuanWorld-Mirror format to ComfyUI format.
    
    Args:
        hwm_images: [1, N, 3, H, W] in range [0, 1]
    
    Returns:
        comfy_images: [B, H, W, C] in range [0, 1]
    """
    # HWM: [1, N, 3, H, W] -> ComfyUI: [B, H, W, C]
    if hwm_images.dim() == 5:
        # [1, N, 3, H, W] -> [N, 3, H, W]
        comfy_images = hwm_images.squeeze(0)
        # [N, 3, H, W] -> [N, H, W, 3]
        comfy_images = comfy_images.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Expected 5D tensor, got {hwm_images.dim()}D")
    
    return comfy_images


def normalize_depth(depth: torch.Tensor, 
                   min_depth: float = None, 
                   max_depth: float = None) -> torch.Tensor:
    """Normalize depth values to [0, 1] range."""
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()
    
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    return depth_norm.clamp(0, 1)


def denormalize_depth(depth_norm: torch.Tensor,
                     min_depth: float,
                     max_depth: float) -> torch.Tensor:
    """Convert normalized depth back to metric depth."""
    return depth_norm * (max_depth - min_depth) + min_depth


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """
    Convert surface normals to RGB visualization.
    
    Args:
        normals: [N, H, W, 3] in range [-1, 1]
    
    Returns:
        rgb: [N, H, W, 3] in range [0, 1]
    """
    # Map from [-1, 1] to [0, 1]
    rgb = (normals + 1.0) / 2.0
    return rgb.clamp(0, 1)
```

### 5.2 Model Loading and Caching

```python
# nodes/loader_nodes.py

import torch
from ..utils.model_cache import ModelCache

class LoadHunyuanWorldMirror:
    """Load and cache HunyuanWorld-Mirror model."""
    
    # Class-level cache
    _cache = ModelCache()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "tencent/HunyuanWorld-Mirror"
                }),
                "device": (["auto", "cuda", "cpu"],),
                "precision": (["fp32", "fp16", "bf16"],),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("HWMIRROR_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanWorld-Mirror/loaders"
    
    def load_model(self, model_name, device, precision, cache_dir=""):
        """Load model with caching."""
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check cache
        cache_key = f"{model_name}_{device}_{precision}"
        if cache_key in self._cache:
            print(f"Loading model from cache: {cache_key}")
            return (self._cache[cache_key],)
        
        # Load model
        print(f"Loading HunyuanWorld-Mirror model...")
        try:
            from src.models.models.worldmirror import WorldMirror
            
            # Set cache directory if provided
            if cache_dir:
                import os
                os.environ['HF_HOME'] = cache_dir
            
            model = WorldMirror.from_pretrained(model_name)
            model = model.to(device)
            
            # Set precision
            if precision == "fp16":
                model = model.half()
            elif precision == "bf16":
                model = model.bfloat16()
            
            model.eval()
            
            # Cache the model
            self._cache[cache_key] = model
            
            print(f"Model loaded successfully on {device} with {precision}")
            return (model,)
            
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            raise
```

### 5.3 Memory Management

```python
# utils/memory_utils.py

import torch
import gc

class MemoryManager:
    """Manage GPU memory for inference."""
    
    @staticmethod
    def clear_cache():
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    @staticmethod
    def get_memory_stats():
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
            }
        return None
    
    @staticmethod
    def estimate_sequence_memory(num_frames, height, width, precision="fp32"):
        """Estimate memory required for a sequence."""
        bytes_per_pixel = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
        }[precision]
        
        # Rough estimate: input + outputs
        input_size = num_frames * height * width * 3 * bytes_per_pixel
        output_size = input_size * 5  # Multiple outputs
        total_size = (input_size + output_size) / 1024**3  # GB
        
        return total_size
    
    @staticmethod
    def check_memory_available(required_gb, safety_margin=2.0):
        """Check if enough GPU memory is available."""
        if not torch.cuda.is_available():
            return False
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        available = total_memory - allocated - safety_margin
        
        return available >= required_gb
```

### 5.4 Export Utilities

```python
# utils/export_utils.py

import numpy as np
import json
from plyfile import PlyData, PlyElement
from typing import Dict, Optional

class ExportUtils:
    """Utilities for exporting 3D data."""
    
    @staticmethod
    def save_point_cloud_ply(filepath: str,
                            points: np.ndarray,
                            colors: Optional[np.ndarray] = None,
                            normals: Optional[np.ndarray] = None):
        """
        Save point cloud to PLY file.
        
        Args:
            filepath: Output file path
            points: [N, 3] array of XYZ coordinates
            colors: [N, 3] array of RGB colors (0-255)
            normals: [N, 3] array of normal vectors
        """
        points = points.reshape(-1, 3)
        
        # Build vertex data
        vertex_data = [
            ('x', points[:, 0]),
            ('y', points[:, 1]),
            ('z', points[:, 2]),
        ]
        
        if colors is not None:
            colors = colors.reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            vertex_data.extend([
                ('red', colors[:, 0]),
                ('green', colors[:, 1]),
                ('blue', colors[:, 2]),
            ])
        
        if normals is not None:
            normals = normals.reshape(-1, 3)
            vertex_data.extend([
                ('nx', normals[:, 0]),
                ('ny', normals[:, 1]),
                ('nz', normals[:, 2]),
            ])
        
        # Create structured array
        vertex_array = np.array(
            list(zip(*[v[1] for v in vertex_data])),
            dtype=[(v[0], v[1].dtype) for v in vertex_data]
        )
        
        # Create PLY element
        el = PlyElement.describe(vertex_array, 'vertex')
        
        # Write file
        PlyData([el]).write(filepath)
    
    @staticmethod
    def save_gaussian_ply(filepath: str,
                         means: np.ndarray,
                         scales: np.ndarray,
                         quats: np.ndarray,
                         colors: np.ndarray,
                         opacities: np.ndarray,
                         sh: Optional[np.ndarray] = None):
        """
        Save 3D Gaussians in standard PLY format.
        
        Args:
            filepath: Output file path
            means: [N, 3] Gaussian centers
            scales: [N, 3] Gaussian scales
            quats: [N, 4] Quaternion rotations (wxyz)
            colors: [N, 3] RGB colors or SH coefficients
            opacities: [N] Opacity values
            sh: [N, SH_COEFFS, 3] Spherical harmonics (optional)
        """
        # Prepare vertex data
        vertex_data = [
            ('x', means[:, 0]),
            ('y', means[:, 1]),
            ('z', means[:, 2]),
            ('scale_0', scales[:, 0]),
            ('scale_1', scales[:, 1]),
            ('scale_2', scales[:, 2]),
            ('rot_0', quats[:, 0]),  # w
            ('rot_1', quats[:, 1]),  # x
            ('rot_2', quats[:, 2]),  # y
            ('rot_3', quats[:, 3]),  # z
            ('opacity', opacities),
        ]
        
        # Add colors or SH coefficients
        if sh is not None:
            # Spherical harmonics
            sh_flat = sh.reshape(len(means), -1)
            for i in range(sh_flat.shape[1]):
                vertex_data.append((f'f_dc_{i}', sh_flat[:, i]))
        else:
            # Simple RGB colors
            vertex_data.extend([
                ('f_dc_0', colors[:, 0]),
                ('f_dc_1', colors[:, 1]),
                ('f_dc_2', colors[:, 2]),
            ])
        
        # Create and save PLY
        vertex_array = np.array(
            list(zip(*[v[1] for v in vertex_data])),
            dtype=[(v[0], v[1].dtype) for v in vertex_data]
        )
        el = PlyElement.describe(vertex_array, 'vertex')
        PlyData([el]).write(filepath)
    
    @staticmethod
    def save_camera_parameters_json(filepath: str,
                                   poses: np.ndarray,
                                   intrinsics: np.ndarray):
        """
        Save camera parameters to JSON.
        
        Args:
            filepath: Output file path
            poses: [N, 4, 4] camera-to-world matrices
            intrinsics: [N, 3, 3] intrinsic matrices
        """
        cameras = []
        for i in range(len(poses)):
            camera = {
                "frame_id": i,
                "pose": poses[i].tolist(),
                "intrinsics": intrinsics[i].tolist(),
                "focal_length": [
                    float(intrinsics[i, 0, 0]),
                    float(intrinsics[i, 1, 1])
                ],
                "principal_point": [
                    float(intrinsics[i, 0, 2]),
                    float(intrinsics[i, 1, 2])
                ],
            }
            cameras.append(camera)
        
        data = {
            "num_frames": len(poses),
            "coordinate_system": "opencv",
            "cameras": cameras,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def save_depth_exr(filepath: str, depth: np.ndarray):
        """
        Save depth map as OpenEXR.
        
        Args:
            filepath: Output file path (.exr)
            depth: [H, W] depth array
        """
        import OpenEXR
        import Imath
        
        height, width = depth.shape
        header = OpenEXR.Header(width, height)
        
        # Convert to float32
        depth_float = depth.astype(np.float32)
        
        # Create EXR file
        exr = OpenEXR.OutputFile(filepath, header)
        exr.writePixels({'Z': depth_float.tobytes()})
        exr.close()
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Test Coverage:**
- Tensor format conversions
- Model loading and caching
- File I/O operations
- Memory management

**Test Framework:** pytest

```python
# tests/test_tensor_utils.py

import torch
import pytest
from utils.tensor_utils import comfy_to_hwmirror, hwmirror_to_comfy

def test_comfy_to_hwmirror():
    # Create sample ComfyUI format tensor
    comfy_tensor = torch.rand(4, 512, 512, 3)  # [B, H, W, C]
    
    # Convert to HWM format
    hwm_tensor = comfy_to_hwmirror(comfy_tensor)
    
    # Verify shape
    assert hwm_tensor.shape == (1, 4, 3, 512, 512)
    
    # Verify values preserved
    assert torch.allclose(
        comfy_tensor[0, :, :, 0],
        hwm_tensor[0, 0, 0, :, :]
    )

def test_round_trip_conversion():
    original = torch.rand(8, 256, 256, 3)
    hwm = comfy_to_hwmirror(original)
    recovered = hwmirror_to_comfy(hwm)
    assert torch.allclose(original, recovered)
```

### 6.2 Integration Tests

**Test Scenarios:**
1. End-to-end inference with sample data
2. Optional prior combinations
3. Variable sequence lengths
4. Memory limits and cleanup
5. Export format validation

### 6.3 Performance Benchmarks

**Metrics to Track:**
- Inference time per sequence length
- Peak GPU memory usage
- Model loading time
- Export operation speeds

**Benchmark Script:**

```python
# tests/benchmark.py

import torch
import time
from nodes.inference_nodes import HunyuanWorldMirrorInference

def benchmark_inference():
    """Benchmark inference performance."""
    
    sequence_lengths = [4, 8, 16, 32]
    results = []
    
    for seq_len in sequence_lengths:
        # Create sample input
        images = torch.rand(seq_len, 512, 512, 3).cuda()
        
        # Time inference
        torch.cuda.synchronize()
        start = time.time()
        
        # Run inference
        # ... inference code ...
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Measure memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        results.append({
            "sequence_length": seq_len,
            "time_seconds": elapsed,
            "peak_memory_gb": peak_memory,
        })
        
        torch.cuda.reset_peak_memory_stats()
    
    return results
```

---

## 7. Documentation Plan

### 7.1 User Documentation

**README.md Contents:**
1. Overview and features
2. Installation instructions (Windows/Linux)
3. Quick start guide
4. Basic workflow examples
5. Troubleshooting
6. FAQ

**Wiki Pages:**
1. Node reference (all nodes documented)
2. Advanced workflows
3. Output format specifications
4. Performance optimization tips
5. Integration with other tools

### 7.2 Example Workflows

**Beginner Workflows:**
1. Simple 3D reconstruction from images
2. Depth map generation
3. Point cloud export

**Advanced Workflows:**
1. Using camera pose priors
2. Gaussian Splatting optimization
3. Novel view synthesis
4. Integration with other ComfyUI nodes

**JSON Workflow Files:**
Provide downloadable ComfyUI workflow files for:
- Basic reconstruction
- Multi-view depth estimation
- 3D Gaussian rendering pipeline

### 7.3 Video Tutorials

**Planned Videos:**
1. Installation walkthrough (5 min)
2. Basic usage tutorial (10 min)
3. Advanced features deep dive (15 min)
4. Troubleshooting common issues (5 min)

---

## 8. Performance Optimization

### 8.1 Memory Optimization

**Strategies:**
1. **Gradient-free inference:** Always use `torch.no_grad()`
2. **Mixed precision:** Support FP16/BF16
3. **Chunked processing:** Process long sequences in chunks
4. **Lazy loading:** Load model only when needed
5. **Garbage collection:** Explicit cleanup after inference

### 8.2 Speed Optimization

**Techniques:**
1. **Model caching:** Cache loaded models
2. **Preprocessed inputs:** Reuse preprocessed data
3. **Batched operations:** Process multiple frames together
4. **CUDA streams:** Async operations where possible
5. **Compiled models:** torch.compile() support

### 8.3 Resource Targets

**Minimum Requirements:**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 16GB system memory
- Storage: 10GB for models and cache

**Recommended Specifications:**
- GPU: NVIDIA RTX 4080 or better (16GB+ VRAM)
- RAM: 32GB system memory
- Storage: SSD with 20GB+ free space

**Performance Targets:**
- 4-frame sequence: < 5 seconds
- 16-frame sequence: < 15 seconds
- Peak memory: < 10GB for typical scenes
- Model loading: < 30 seconds (first time)

---

## 9. Maintenance and Updates

### 9.1 Version Control

**Semantic Versioning:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking changes to node interfaces
- **MINOR:** New nodes or features
- **PATCH:** Bug fixes and optimizations

### 9.2 Update Strategy

**Release Schedule:**
- Patch releases: As needed for critical bugs
- Minor releases: Monthly with new features
- Major releases: Quarterly or for significant changes

**Changelog Maintenance:**
- Document all changes in CHANGELOG.md
- Include migration guides for breaking changes
- Note compatibility with ComfyUI versions

### 9.3 Community Support

**Support Channels:**
1. GitHub Issues (bug reports, feature requests)
2. GitHub Discussions (questions, showcases)
3. Discord server (real-time help)
4. Documentation wiki (tutorials, guides)

**Response Time Goals:**
- Critical bugs: 24 hours
- Feature requests: 1 week initial response
- Questions: 48 hours

---

## 10. Future Enhancements

### 10.1 Short-term (3 months)

1. **Additional output formats:**
   - USD export for 3D software
   - FBX/OBJ with materials
   - NeRF format compatibility

2. **Batch processing:**
   - Process multiple sequences
   - Queue management

3. **Quality presets:**
   - Fast/balanced/quality modes
   - Auto-adjust based on hardware

### 10.2 Medium-term (6 months)

1. **Advanced features:**
   - Mesh reconstruction from point clouds
   - Texture mapping and unwrapping
   - Multi-resolution processing

2. **Integration features:**
   - ControlNet compatibility
   - Depth conditioning for Stable Diffusion
   - Camera path animation

3. **Optimization:**
   - TensorRT acceleration
   - Model quantization
   - Mobile/web deployment

### 10.3 Long-term (12 months)

1. **Extended capabilities:**
   - Video-to-3D conversion
   - Dynamic scene reconstruction
   - Real-time preview

2. **Training support:**
   - Fine-tuning on custom data
   - Domain adaptation
   - Few-shot learning

3. **Ecosystem integration:**
   - Blender plugin
   - Unreal Engine integration
   - Unity compatibility

---

## 11. Risk Assessment

### 11.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| CUDA compatibility issues | High | Medium | Pre-compiled wheels, fallback options |
| Memory overflow on large scenes | High | Medium | Chunked processing, memory monitoring |
| Model download failures | Medium | Low | Local caching, retry logic |
| Tensor format bugs | High | Medium | Comprehensive unit tests |
| Performance degradation | Medium | Low | Benchmarking, optimization |

### 11.2 User Experience Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Complex installation | High | Medium | Automated scripts, clear docs |
| Confusing node connections | Medium | Medium | Example workflows, tutorials |
| Poor error messages | Medium | High | User-friendly error handling |
| Lack of documentation | High | Low | Comprehensive docs from day 1 |

### 11.3 Maintenance Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Upstream model changes | High | Low | Version pinning, migration guides |
| ComfyUI API changes | Medium | Medium | Monitor upstream, quick updates |
| Dependency conflicts | Medium | Medium | Locked requirements, testing |
| Community support burden | Medium | High | Good docs, active community |

---

## 12. Success Metrics

### 12.1 Adoption Metrics

- **Downloads:** Target 1000+ in first 3 months
- **GitHub stars:** Target 500+ in first 6 months
- **Active users:** Target 100+ monthly active
- **Community showcase:** Target 50+ user projects

### 12.2 Quality Metrics

- **Bug reports:** < 5 critical bugs in first month
- **Installation success:** > 90% successful installs
- **User satisfaction:** > 4.0/5.0 rating
- **Performance:** Meet all performance targets

### 12.3 Development Metrics

- **Test coverage:** > 80% code coverage
- **Documentation:** 100% of nodes documented
- **Response time:** < 48 hours average
- **Release cadence:** Monthly minor releases

---

## 13. Conclusion

This design document provides a comprehensive blueprint for developing a ComfyUI node pack for HunyuanWorld-Mirror. The modular architecture, clear node specifications, and thorough testing strategy ensure a robust and user-friendly implementation.

**Key Success Factors:**
1. Easy Windows installation with pre-compiled binaries
2. Intuitive node design following ComfyUI conventions
3. Comprehensive documentation and examples
4. Efficient memory and performance optimization
5. Active community support and maintenance

**Next Steps:**
1. Review and approve design document
2. Set up development environment
3. Implement core loader and inference nodes
4. Create basic example workflows
5. Begin alpha testing with early adopters

---

## Appendix A: Related Resources

**Official Documentation:**
- HunyuanWorld-Mirror: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
- gsplat: https://github.com/nerfstudio-project/gsplat
- ComfyUI: https://github.com/comfyanonymous/ComfyUI

**Tutorials:**
- gsplat Windows Installation: https://www.youtube.com/watch?v=ACPTiP98Pf8
- ComfyUI Custom Nodes: https://docs.comfy.org/essentials/custom_node_tutorial

**Community:**
- HunyuanWorld Discord: https://discord.gg/hunyuanworld
- ComfyUI Discord: https://discord.gg/comfyui
- gsplat Discussions: https://github.com/nerfstudio-project/gsplat/discussions

---

## Appendix B: Glossary

- **3DGS:** 3D Gaussian Splatting - A method for representing 3D scenes
- **ComfyUI:** Node-based UI for Stable Diffusion and AI workflows
- **HWM:** HunyuanWorld-Mirror (abbreviation)
- **PLY:** Polygon File Format - Common 3D data format
- **Prior:** Additional input information to guide the model
- **Splat:** Individual Gaussian primitive in 3D Gaussian Splatting
- **VRAM:** Video RAM - GPU memory

---

**Document Version:** 1.0  
**Last Updated:** November 5, 2025  
**Author:** Development Team  
**Status:** Ready for Implementation
