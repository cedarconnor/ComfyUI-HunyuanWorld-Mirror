"""
Tensor format conversion utilities for HunyuanWorld-Mirror ComfyUI integration.

Handles conversion between ComfyUI and HunyuanWorld-Mirror tensor formats.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def comfy_to_hwm(images: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """
    Convert ComfyUI image format to HunyuanWorld-Mirror format.

    Args:
        images: ComfyUI format tensor [B, H, W, C] in range [0, 1]
        patch_size: Model patch size (default 14). Images will be resized to multiples of this.

    Returns:
        HWM format tensor [1, N, 3, H, W] in range [0, 1]

    Example:
        >>> comfy_images = torch.rand(4, 512, 512, 3)  # 4 images
        >>> hwm_images = comfy_to_hwm(comfy_images)
        >>> hwm_images.shape
        torch.Size([1, 4, 3, 512, 512])
    """
    import torch.nn.functional as F

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(images)}")

    if images.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B, H, W, C], got {images.dim()}D tensor")

    B, H, W, C = images.shape

    if C != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {C} channels")

    # Calculate target dimensions (nearest multiples of patch_size)
    target_h = round(H / patch_size) * patch_size
    target_w = round(W / patch_size) * patch_size

    # Resize if needed
    if H != target_h or W != target_w:
        print(f"[INFO] Resizing images from {H}x{W} to {target_h}x{target_w} (multiple of {patch_size})")
        # Permute to [B, C, H, W] for resize
        images_chw = images.permute(0, 3, 1, 2)
        images_resized = F.interpolate(images_chw, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # Permute back to [B, H, W, C]
        images = images_resized.permute(0, 2, 3, 1)

    # [B, H, W, C] -> [B, C, H, W] -> [1, B, C, H, W]
    hwm_images = images.permute(0, 3, 1, 2).unsqueeze(0)

    return hwm_images


def hwm_to_comfy(images: torch.Tensor) -> torch.Tensor:
    """
    Convert HunyuanWorld-Mirror format to ComfyUI format.

    Args:
        images: HWM format tensor [1, N, 3, H, W] in range [0, 1]

    Returns:
        ComfyUI format tensor [B, H, W, C] in range [0, 1]

    Example:
        >>> hwm_images = torch.rand(1, 4, 3, 512, 512)
        >>> comfy_images = hwm_to_comfy(hwm_images)
        >>> comfy_images.shape
        torch.Size([4, 512, 512, 3])
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(images)}")

    if images.dim() != 5:
        raise ValueError(f"Expected 5D tensor [1, N, 3, H, W], got {images.dim()}D tensor")

    # [1, N, 3, H, W] -> [N, 3, H, W] -> [N, H, W, 3]
    comfy_images = images.squeeze(0).permute(0, 2, 3, 1)

    return comfy_images


def normalize_depth(
    depth: torch.Tensor,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Normalize depth values to [0, 1] range.

    Args:
        depth: Depth tensor of any shape
        min_depth: Minimum depth value (if None, use tensor min)
        max_depth: Maximum depth value (if None, use tensor max)
        epsilon: Small value to avoid division by zero

    Returns:
        Normalized depth tensor in range [0, 1]
    """
    if min_depth is None:
        min_depth = depth.min().item()
    if max_depth is None:
        max_depth = depth.max().item()

    # Normalize to [0, 1]
    depth_norm = (depth - min_depth) / (max_depth - min_depth + epsilon)
    depth_norm = depth_norm.clamp(0, 1)

    return depth_norm


def denormalize_depth(
    depth_norm: torch.Tensor,
    min_depth: float,
    max_depth: float
) -> torch.Tensor:
    """
    Convert normalized depth back to metric depth.

    Args:
        depth_norm: Normalized depth in range [0, 1]
        min_depth: Original minimum depth value
        max_depth: Original maximum depth value

    Returns:
        Metric depth tensor
    """
    depth = depth_norm * (max_depth - min_depth) + min_depth
    return depth


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """
    Convert surface normals to RGB visualization.

    Maps normal vectors from [-1, 1] to RGB colors [0, 1].
    Standard visualization: X→Red, Y→Green, Z→Blue

    Args:
        normals: Normal tensor [..., 3] in range [-1, 1]

    Returns:
        RGB tensor [..., 3] in range [0, 1]

    Example:
        >>> normals = torch.randn(4, 512, 512, 3)
        >>> normals = normals / normals.norm(dim=-1, keepdim=True)  # Normalize
        >>> rgb = normals_to_rgb(normals)
        >>> rgb.min() >= 0 and rgb.max() <= 1
        True
    """
    # Map from [-1, 1] to [0, 1]
    rgb = (normals + 1.0) / 2.0
    rgb = rgb.clamp(0, 1)

    return rgb


def rgb_to_normals(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB visualization back to surface normals.

    Maps RGB colors from [0, 1] to normal vectors [-1, 1].

    Args:
        rgb: RGB tensor [..., 3] in range [0, 1]

    Returns:
        Normal tensor [..., 3] in range [-1, 1]
    """
    # Map from [0, 1] to [-1, 1]
    normals = rgb * 2.0 - 1.0

    # Normalize to unit vectors
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

    return normals


def validate_image_tensor(images: torch.Tensor, name: str = "images") -> None:
    """
    Validate that a tensor is a valid image tensor for ComfyUI.

    Args:
        images: Tensor to validate
        name: Name for error messages

    Raises:
        TypeError: If not a torch.Tensor
        ValueError: If shape or value range is invalid
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name}: Expected torch.Tensor, got {type(images)}")

    if images.dim() != 4:
        raise ValueError(f"{name}: Expected 4D tensor [B, H, W, C], got {images.dim()}D")

    B, H, W, C = images.shape

    if C != 3:
        raise ValueError(f"{name}: Expected 3 channels (RGB), got {C}")

    if images.min() < -0.1 or images.max() > 1.1:
        print(f"Warning: {name} values outside [0, 1] range: [{images.min():.3f}, {images.max():.3f}]")


def resize_depth(
    depth: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Resize depth map to target size.

    Args:
        depth: Depth tensor [B, H, W] or [B, 1, H, W]
        target_size: (height, width)
        mode: Interpolation mode ('nearest', 'bilinear')

    Returns:
        Resized depth tensor
    """
    import torch.nn.functional as F

    if depth.dim() == 3:
        depth = depth.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        squeeze = True
    else:
        squeeze = False

    resized = F.interpolate(
        depth,
        size=target_size,
        mode=mode,
        align_corners=False if mode == 'bilinear' else None
    )

    if squeeze:
        resized = resized.squeeze(1)

    return resized


def compute_point_cloud_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute 3D point cloud from depth map and camera parameters.

    Args:
        depth: Depth map [H, W]
        intrinsics: Camera intrinsic matrix [3, 3]
        extrinsics: Camera extrinsic matrix [4, 4] (optional, for world coords)

    Returns:
        Point cloud [H, W, 3] in camera or world coordinates
    """
    H, W = depth.shape

    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing='ij'
    )

    # Get camera parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Back-project to 3D (camera coordinates)
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    # Stack to [H, W, 3]
    points_cam = torch.stack([X, Y, Z], dim=-1)

    # Transform to world coordinates if extrinsics provided
    if extrinsics is not None:
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]

        # Reshape for matrix multiplication
        points_flat = points_cam.reshape(-1, 3)
        points_world = (R @ points_flat.T).T + t
        points_cam = points_world.reshape(H, W, 3)

    return points_cam
