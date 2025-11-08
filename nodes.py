"""
HunyuanWorld-Mirror ComfyUI Node Pack - Main Nodes

All 8 core nodes for 3D reconstruction from images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, Dict, Any, Optional

from .utils import (
    comfy_to_hwm,
    hwm_to_comfy,
    normalize_depth,
    normals_to_rgb,
    MemoryManager,
    ModelCache,
    InferenceWrapper,
    ExportUtils,
    tensor_to_numpy,
)
from .utils.inference import load_model


# ============================================================================
# Node 1: LoadHunyuanWorldMirrorModel
# ============================================================================

class LoadHunyuanWorldMirrorModel:
    """
    Load the HunyuanWorld-Mirror model with automatic caching.

    This node loads the model once and caches it for reuse across workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "HunyuanWorld-Mirror",
                    "multiline": False,
                    "tooltip": "Model name, filename, or path. Checks ComfyUI/models/HunyuanWorld-Mirror/ first. Examples: 'HunyuanWorld-Mirror', 'model.safetensors', or full path."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Which device to run the model on. 'auto' selects CUDA if available, otherwise CPU. Use 'cuda' for GPU acceleration (recommended) or 'cpu' for compatibility."
                }),
                "precision": (["fp32", "fp16", "bf16"], {
                    "default": "fp16",
                    "tooltip": "Numeric precision for model weights. fp16 (half precision) uses less memory and is faster, fp32 (full precision) is more accurate, bf16 (bfloat16) balances both on supported GPUs."
                }),
            },
            "optional": {
                "cache_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom directory for downloading and caching model files from HuggingFace. Leave empty to use default cache location."
                }),
            }
        }

    RETURN_TYPES = ("HWMIRROR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanWorld-Mirror/loaders"

    def load_model(
        self,
        model_name: str,
        device: str,
        precision: str,
        cache_dir: str = ""
    ) -> Tuple[Any]:
        """Load and cache the model."""

        print("=" * 60)
        print("Loading HunyuanWorld-Mirror Model")
        print("=" * 60)

        try:
            # Load model (with caching)
            model, cache_key = load_model(
                model_name=model_name,
                device=device,
                precision=precision,
                cache_dir=cache_dir if cache_dir else None,
                use_cache=True
            )

            print(f"Model ready: {cache_key}")
            print("=" * 60)

            return (model,)

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise


# ============================================================================
# Node 2: HWMInference
# ============================================================================

class HWMInference:
    """
    Main inference node - generates all 3D outputs in a single pass.

    Outputs: depth, normals, points3d, camera_poses, camera_intrinsics, gaussians
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HWMIRROR_MODEL", {
                    "tooltip": "The loaded HunyuanWorld-Mirror model from the 'Load HunyuanWorld-Mirror Model' node."
                }),
                "images": ("IMAGE", {
                    "tooltip": "Sequence of input images (4-64 frames). Use LoadImage + ImageBatch to create a sequence. More frames give better 3D reconstruction but use more memory."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducible results. Set to -1 for random seed each time, or use a specific number (e.g., 42) to get the same results every run."
                }),
            },
        }

    RETURN_TYPES = ("DEPTH", "NORMALS", "POINTS3D", "POSES", "INTRINSICS", "GAUSSIANS")
    RETURN_NAMES = ("depth", "normals", "points3d", "camera_poses", "camera_intrinsics", "gaussians")
    FUNCTION = "infer"
    CATEGORY = "HunyuanWorld-Mirror/inference"

    def infer(
        self,
        model: Any,
        images: torch.Tensor,
        seed: int
    ) -> Tuple:
        """Run inference on image sequence."""

        print("\n" + "=" * 60)
        print("HunyuanWorld-Mirror Inference")
        print("=" * 60)

        # Set seed if provided
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Get input info
        B, H, W, C = images.shape
        print(f"Input: {B} images at {H}x{W}x{C}")

        # Estimate memory
        device = next(model.parameters()).device
        precision = "fp16" if next(model.parameters()).dtype == torch.float16 else "fp32"

        estimated_mem = MemoryManager.estimate_sequence_memory(B, H, W, precision)
        print(f"Estimated memory: {estimated_mem:.2f}GB")

        # Check memory
        available, msg = MemoryManager.check_memory_available(estimated_mem)
        if not available:
            print(f"Warning: {msg}")

        # Convert to HWM format
        print("Converting to HWM format...")
        hwm_images = comfy_to_hwm(images)  # [B, H, W, C] -> [1, N, 3, H, W]

        # Run inference
        print("Running inference...")
        MemoryManager.print_memory_stats("Before inference - ")

        try:
            with torch.no_grad():
                # Create inference wrapper
                wrapper = InferenceWrapper(model, str(device), precision)

                # Run inference (no conditions for basic mode)
                outputs = wrapper.infer(hwm_images, condition=None)

            MemoryManager.print_memory_stats("After inference - ")

            # Extract outputs
            # Note: Actual output keys depend on the real model implementation
            # These are based on the design doc specifications
            depth = outputs.get('depth', outputs.get('pred_depth', None))
            normals = outputs.get('normals', outputs.get('pred_normals', None))
            pts3d = outputs.get('pts3d', outputs.get('pred_pts3d', None))
            poses = outputs.get('camera_poses', outputs.get('pred_poses', None))
            intrinsics = outputs.get('camera_intrinsics', outputs.get('pred_intrinsics', None))

            # Extract Gaussian parameters
            gaussian_params = {
                'means': outputs.get('gaussian_means', None),
                'scales': outputs.get('gaussian_scales', None),
                'quats': outputs.get('gaussian_quats', None),
                'colors': outputs.get('gaussian_colors', None),
                'opacities': outputs.get('gaussian_opacities', None),
            }

            print("✓ Inference complete")
            print(f"  Depth: {depth.shape if depth is not None else 'N/A'}")
            print(f"  Normals: {normals.shape if normals is not None else 'N/A'}")
            print(f"  Points3D: {pts3d.shape if pts3d is not None else 'N/A'}")
            print(f"  Poses: {poses.shape if poses is not None else 'N/A'}")
            print(f"  Intrinsics: {intrinsics.shape if intrinsics is not None else 'N/A'}")
            print("=" * 60)

            # Clear memory
            wrapper.clear_memory()

            return (depth, normals, pts3d, poses, intrinsics, gaussian_params)

        except Exception as e:
            print(f"✗ Inference error: {e}")
            MemoryManager.clear_cache()
            raise


# ============================================================================
# Node 3: VisualizeDepth
# ============================================================================

class VisualizeDepth:
    """
    Convert depth maps to colorized visualizations for preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("DEPTH", {
                    "tooltip": "Depth map output from the HWM Inference node. Contains distance information for each pixel in the image."
                }),
                "colormap": (["viridis", "plasma", "turbo", "magma", "inferno", "gray"], {
                    "default": "turbo",
                    "tooltip": "Color scheme for visualizing depth. 'turbo' is rainbow-like (blue=close, red=far), 'viridis' is blue-to-yellow, 'gray' is grayscale. Choose based on preference."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to automatically adjust depth values to use the full color range. Enable (True) for better visualization, disable (False) to use raw depth values."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "HunyuanWorld-Mirror/visualization"

    def visualize(
        self,
        depth: torch.Tensor,
        colormap: str,
        normalize: bool
    ) -> Tuple[torch.Tensor]:
        """Convert depth to colored image."""

        # Convert to numpy
        depth_np = tensor_to_numpy(depth)

        # Handle batch dimension
        if depth_np.ndim == 3:
            # [N, H, W]
            batch_size = depth_np.shape[0]
        elif depth_np.ndim == 4:
            # [1, N, H, W]
            depth_np = depth_np.squeeze(0)
            batch_size = depth_np.shape[0]
        else:
            raise ValueError(f"Unexpected depth shape: {depth_np.shape}")

        # Process each depth map
        colored_images = []
        cmap = cm.get_cmap(colormap)

        for i in range(batch_size):
            depth_single = depth_np[i]

            # Normalize
            if normalize:
                d_min = depth_single.min()
                d_max = depth_single.max()
                depth_norm = (depth_single - d_min) / (d_max - d_min + 1e-8)
            else:
                depth_norm = depth_single

            # Apply colormap
            colored = cmap(depth_norm)  # Returns RGBA
            colored_rgb = colored[:, :, :3]  # Take RGB only

            colored_images.append(colored_rgb)

        # Stack and convert to torch tensor
        images_np = np.stack(colored_images, axis=0)  # [N, H, W, 3]
        images_torch = torch.from_numpy(images_np).float()

        print(f"✓ Visualized {batch_size} depth maps with '{colormap}' colormap")

        return (images_torch,)


# ============================================================================
# Node 4: VisualizeNormals
# ============================================================================

class VisualizeNormals:
    """
    Convert surface normals to RGB visualization.
    Standard mapping: X→Red, Y→Green, Z→Blue
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normals": ("NORMALS", {
                    "tooltip": "Surface normal vectors from the HWM Inference node. Shows which direction each surface is facing in 3D space. Converted to RGB where X=Red, Y=Green, Z=Blue."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "HunyuanWorld-Mirror/visualization"

    def visualize(self, normals: torch.Tensor) -> Tuple[torch.Tensor]:
        """Convert normals to RGB image."""

        # Normals are expected in range [-1, 1]
        # Convert to [0, 1] for visualization
        normals_rgb = normals_to_rgb(normals)

        print(f"✓ Visualized normals: {normals_rgb.shape}")

        return (normals_rgb,)


# ============================================================================
# Node 5: SavePointCloud
# ============================================================================

class SavePointCloud:
    """
    Export 3D point cloud to standard formats.
    Supports: PLY, PCD, OBJ, XYZ
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points3d": ("POINTS3D", {
                    "tooltip": "3D point coordinates from the HWM Inference node. Each point represents a location in 3D space (X, Y, Z coordinates)."
                }),
                "filepath": ("STRING", {
                    "default": "./output/pointcloud.ply",
                    "multiline": False,
                    "tooltip": "Where to save the point cloud file. Can be relative (./output/file.ply) or absolute path (C:/Users/Name/Documents/file.ply). File extension will auto-adjust to match format."
                }),
                "format": (["ply", "obj", "xyz"], {
                    "default": "ply",
                    "tooltip": "File format for the point cloud. PLY is most common and supports colors/normals. OBJ works with most 3D software. XYZ is simple text format (just coordinates)."
                }),
            },
            "optional": {
                "colors": ("IMAGE", {
                    "tooltip": "Optional: RGB colors for each point, typically from the source images. Makes the point cloud look more realistic when viewed in 3D software."
                }),
                "normals": ("NORMALS", {
                    "tooltip": "Optional: Surface normal directions for each point. Helps with lighting and rendering in 3D viewers. Only supported in PLY format."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        points3d: torch.Tensor,
        filepath: str,
        format: str,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None
    ) -> Tuple[str]:
        """Save point cloud to file."""

        # Convert to numpy
        points_np = tensor_to_numpy(points3d)
        colors_np = tensor_to_numpy(colors) if colors is not None else None
        normals_np = tensor_to_numpy(normals) if normals is not None else None

        # Ensure file extension matches format
        if not filepath.endswith(f'.{format}'):
            filepath = filepath.rsplit('.', 1)[0] + f'.{format}'

        # Save based on format
        if format == "ply":
            saved_path = ExportUtils.save_point_cloud_ply(
                filepath, points_np, colors_np, normals_np
            )
        elif format == "obj":
            saved_path = ExportUtils.save_point_cloud_obj(
                filepath, points_np, colors_np
            )
        elif format == "xyz":
            saved_path = ExportUtils.save_point_cloud_xyz(
                filepath, points_np, colors_np
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        return (saved_path,)


# ============================================================================
# Node 6: Save3DGaussians
# ============================================================================

class Save3DGaussians:
    """
    Export 3D Gaussian Splatting representation.
    Standard 3DGS PLY format compatible with viewers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gaussians": ("GAUSSIANS", {
                    "tooltip": "3D Gaussian Splatting parameters from the HWM Inference node. Contains position, scale, rotation, color, and opacity for each Gaussian primitive."
                }),
                "filepath": ("STRING", {
                    "default": "./output/gaussians.ply",
                    "multiline": False,
                    "tooltip": "Where to save the Gaussian Splatting file. Use .ply extension. This file can be loaded in 3DGS viewers for real-time novel view synthesis."
                }),
                "include_sh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Whether to include Spherical Harmonics coefficients for view-dependent appearance. Enable for more realistic lighting effects, disable for smaller files and faster loading."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        gaussians: Dict[str, torch.Tensor],
        filepath: str,
        include_sh: bool
    ) -> Tuple[str]:
        """Save Gaussian parameters to PLY file."""

        # Extract parameters
        means = tensor_to_numpy(gaussians['means'])
        scales = tensor_to_numpy(gaussians['scales'])
        quats = tensor_to_numpy(gaussians['quats'])
        colors = tensor_to_numpy(gaussians['colors'])
        opacities = tensor_to_numpy(gaussians['opacities'])

        sh = None
        if include_sh and 'sh' in gaussians:
            sh = tensor_to_numpy(gaussians['sh'])

        # Ensure .ply extension
        if not filepath.endswith('.ply'):
            filepath = filepath.rsplit('.', 1)[0] + '.ply'

        # Save
        saved_path = ExportUtils.save_gaussian_ply(
            filepath, means, scales, quats, colors, opacities, sh
        )

        return (saved_path,)


# ============================================================================
# Node 7: SaveDepthMap
# ============================================================================

class SaveDepthMap:
    """
    Export depth maps in various precision formats.
    Supports: NPY, EXR, PFM, PNG16
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("DEPTH", {
                    "tooltip": "Depth map from the HWM Inference node. Raw depth values will be saved with full precision."
                }),
                "filepath": ("STRING", {
                    "default": "./output/depth.npy",
                    "multiline": False,
                    "tooltip": "Where to save the depth data. File extension will auto-adjust to match the selected format (e.g., depth.npy, depth.exr, depth.png)."
                }),
                "format": (["npy", "exr", "png16"], {
                    "default": "npy",
                    "tooltip": "File format for depth data. NPY is NumPy binary (full precision, Python-friendly). EXR is OpenEXR (high dynamic range, used in VFX). PNG16 is 16-bit PNG (good compatibility)."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        depth: torch.Tensor,
        filepath: str,
        format: str
    ) -> Tuple[str]:
        """Save depth map to file."""

        # Convert to numpy
        depth_np = tensor_to_numpy(depth)

        # Ensure correct extension
        if not filepath.endswith(f'.{format}'):
            filepath = filepath.rsplit('.', 1)[0] + f'.{format}'

        # Save based on format
        if format == "npy":
            saved_path = ExportUtils.save_depth_npy(filepath, depth_np)
        elif format == "exr":
            saved_path = ExportUtils.save_depth_exr(filepath, depth_np)
        elif format == "png16":
            saved_path = ExportUtils.save_depth_png16(filepath, depth_np)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return (saved_path,)


# ============================================================================
# Node 8: SaveCameraParams
# ============================================================================

class SaveCameraParams:
    """
    Export camera parameters for 3D reconstruction tools.
    Supports: JSON, COLMAP, NPY
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_poses": ("POSES", {
                    "tooltip": "Camera pose matrices from the HWM Inference node. Each pose is a 4x4 matrix describing camera position and orientation in 3D space for each frame."
                }),
                "camera_intrinsics": ("INTRINSICS", {
                    "tooltip": "Camera intrinsic parameters from the HWM Inference node. 3x3 matrix containing focal length, principal point, and other internal camera properties."
                }),
                "filepath": ("STRING", {
                    "default": "./output/cameras.json",
                    "multiline": False,
                    "tooltip": "Where to save camera parameters. JSON format saves in one readable file. NPY format saves as two files: filepath_poses.npy and filepath_intrinsics.npy."
                }),
                "format": (["json", "npy"], {
                    "default": "json",
                    "tooltip": "File format for camera data. JSON is human-readable and good for debugging. NPY is binary format for fast loading in Python/NumPy scripts."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "HunyuanWorld-Mirror/output"
    OUTPUT_NODE = True

    def save(
        self,
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        filepath: str,
        format: str
    ) -> Tuple[str]:
        """Save camera parameters to file."""

        # Convert to numpy
        poses_np = tensor_to_numpy(camera_poses)
        intrinsics_np = tensor_to_numpy(camera_intrinsics)

        # Save based on format
        if format == "json":
            if not filepath.endswith('.json'):
                filepath = filepath.rsplit('.', 1)[0] + '.json'
            saved_path = ExportUtils.save_camera_parameters_json(
                filepath, poses_np, intrinsics_np
            )
        elif format == "npy":
            # Remove extension for NPY (will add _poses.npy and _intrinsics.npy)
            base_path = filepath.rsplit('.', 1)[0]
            saved_path = ExportUtils.save_camera_parameters_npy(
                base_path, poses_np, intrinsics_np
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        return (saved_path,)


# ============================================================================
# Node Mappings for ComfyUI Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadHunyuanWorldMirrorModel": LoadHunyuanWorldMirrorModel,
    "HWMInference": HWMInference,
    "VisualizeDepth": VisualizeDepth,
    "VisualizeNormals": VisualizeNormals,
    "SavePointCloud": SavePointCloud,
    "Save3DGaussians": Save3DGaussians,
    "SaveDepthMap": SaveDepthMap,
    "SaveCameraParams": SaveCameraParams,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadHunyuanWorldMirrorModel": "Load HunyuanWorld-Mirror Model",
    "HWMInference": "HWM Inference",
    "VisualizeDepth": "Visualize Depth",
    "VisualizeNormals": "Visualize Normals",
    "SavePointCloud": "Save Point Cloud",
    "Save3DGaussians": "Save 3D Gaussians",
    "SaveDepthMap": "Save Depth Map",
    "SaveCameraParams": "Save Camera Parameters",
}
