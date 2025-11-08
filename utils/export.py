"""
Export utilities for 3D data formats.

Handles exporting point clouds, Gaussian splats, depth maps, and camera parameters
to various standard formats.
"""

import numpy as np
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Optional imports - gracefully handle missing dependencies
try:
    from plyfile import PlyData, PlyElement
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    print("Warning: plyfile not installed. PLY export will not be available.")
    print("Install with: pip install plyfile")


class ExportUtils:
    """Utilities for exporting 3D data to various formats."""

    # ============================================================================
    # Point Cloud Export
    # ============================================================================

    @staticmethod
    def save_point_cloud_ply(
        filepath: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        binary: bool = True
    ) -> str:
        """
        Save point cloud to PLY file.

        Args:
            filepath: Output file path (.ply)
            points: Point positions [N, 3] (XYZ)
            colors: Point colors [N, 3] (RGB in range [0, 1] or [0, 255])
            normals: Point normals [N, 3] (normalized vectors)
            binary: Save in binary format (more compact)

        Returns:
            filepath: Path to saved file
        """
        if not PLYFILE_AVAILABLE:
            raise ImportError(
                "plyfile is required for PLY export. "
                "Install with: pip install plyfile"
            )

        # Ensure output directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Flatten points
        points = points.reshape(-1, 3).astype(np.float32)
        num_points = len(points)

        # Build vertex data list
        vertex_data = [
            ('x', points[:, 0]),
            ('y', points[:, 1]),
            ('z', points[:, 2]),
        ]

        # Add colors if provided
        if colors is not None:
            colors = colors.reshape(-1, 3)
            # Convert to 0-255 range if needed
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)

            vertex_data.extend([
                ('red', colors[:, 0]),
                ('green', colors[:, 1]),
                ('blue', colors[:, 2]),
            ])

        # Add normals if provided
        if normals is not None:
            normals = normals.reshape(-1, 3).astype(np.float32)
            vertex_data.extend([
                ('nx', normals[:, 0]),
                ('ny', normals[:, 1]),
                ('nz', normals[:, 2]),
            ])

        # Create structured array
        dtype = [(name, arr.dtype) for name, arr in vertex_data]
        vertex_array = np.zeros(num_points, dtype=dtype)
        for name, arr in vertex_data:
            vertex_array[name] = arr

        # Create PLY element
        el = PlyElement.describe(vertex_array, 'vertex')

        # Write file
        PlyData([el], text=not binary).write(filepath)

        print(f"✓ Saved point cloud: {filepath} ({num_points} points)")
        return filepath

    @staticmethod
    def save_point_cloud_obj(
        filepath: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ) -> str:
        """
        Save point cloud to OBJ file.

        Args:
            filepath: Output file path (.obj)
            points: Point positions [N, 3]
            colors: Point colors [N, 3] (optional, ignored in OBJ)

        Returns:
            filepath: Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        points = points.reshape(-1, 3)

        with open(filepath, 'w') as f:
            f.write("# Point Cloud exported from HunyuanWorld-Mirror\n")
            for pt in points:
                f.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        print(f"✓ Saved point cloud: {filepath} ({len(points)} points)")
        return filepath

    @staticmethod
    def save_point_cloud_xyz(
        filepath: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ) -> str:
        """
        Save point cloud to XYZ file (simple ASCII format).

        Args:
            filepath: Output file path (.xyz)
            points: Point positions [N, 3]
            colors: Point colors [N, 3] (RGB in range [0, 1])

        Returns:
            filepath: Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        points = points.reshape(-1, 3)

        with open(filepath, 'w') as f:
            if colors is not None:
                colors = colors.reshape(-1, 3)
                # Ensure colors in [0, 1] range
                if colors.max() > 1.0:
                    colors = colors / 255.0

                for pt, col in zip(points, colors):
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                           f"{col[0]:.4f} {col[1]:.4f} {col[2]:.4f}\n")
            else:
                for pt in points:
                    f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")

        print(f"✓ Saved point cloud: {filepath} ({len(points)} points)")
        return filepath

    # ============================================================================
    # 3D Gaussian Splatting Export
    # ============================================================================

    @staticmethod
    def save_gaussian_ply(
        filepath: str,
        means: np.ndarray,
        scales: np.ndarray,
        quats: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
        sh: Optional[np.ndarray] = None
    ) -> str:
        """
        Save 3D Gaussians in standard 3DGS PLY format.

        Args:
            filepath: Output file path (.ply)
            means: Gaussian centers [N, 3]
            scales: Gaussian scales [N, 3]
            quats: Quaternion rotations [N, 4] (wxyz order)
            colors: RGB colors [N, 3] or SH coefficients
            opacities: Opacity values [N, 1] or [N]
            sh: Spherical harmonics [N, SH_COEFFS, 3] (optional)

        Returns:
            filepath: Path to saved file
        """
        if not PLYFILE_AVAILABLE:
            raise ImportError(
                "plyfile is required for 3DGS PLY export. "
                "Install with: pip install plyfile"
            )

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        num_gaussians = len(means)

        # Ensure correct shapes
        means = means.reshape(-1, 3).astype(np.float32)
        scales = scales.reshape(-1, 3).astype(np.float32)
        quats = quats.reshape(-1, 4).astype(np.float32)
        colors = colors.reshape(-1, 3).astype(np.float32)

        if opacities.ndim == 1:
            opacities = opacities.reshape(-1, 1)
        opacities = opacities.astype(np.float32)

        # Build vertex data
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
            ('opacity', opacities[:, 0]),
        ]

        # Add colors or spherical harmonics
        if sh is not None:
            # Flatten SH coefficients
            sh_flat = sh.reshape(num_gaussians, -1).astype(np.float32)
            for i in range(sh_flat.shape[1]):
                vertex_data.append((f'f_dc_{i}', sh_flat[:, i]))
        else:
            # Simple RGB colors
            vertex_data.extend([
                ('f_dc_0', colors[:, 0]),
                ('f_dc_1', colors[:, 1]),
                ('f_dc_2', colors[:, 2]),
            ])

        # Create structured array
        dtype = [(name, arr.dtype) for name, arr in vertex_data]
        vertex_array = np.zeros(num_gaussians, dtype=dtype)
        for name, arr in vertex_data:
            vertex_array[name] = arr

        # Create PLY element
        el = PlyElement.describe(vertex_array, 'vertex')

        # Write file (always binary for 3DGS)
        PlyData([el], text=False).write(filepath)

        print(f"✓ Saved 3D Gaussians: {filepath} ({num_gaussians} Gaussians)")
        return filepath

    # ============================================================================
    # Depth Map Export
    # ============================================================================

    @staticmethod
    def save_depth_npy(filepath: str, depth: np.ndarray) -> str:
        """Save depth map as NumPy array."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, depth.astype(np.float32))
        print(f"✓ Saved depth map: {filepath}")
        return filepath

    @staticmethod
    def save_depth_exr(filepath: str, depth: np.ndarray) -> str:
        """
        Save depth map as OpenEXR file.

        Requires OpenEXR library. Falls back to NPY if not available.
        """
        try:
            import OpenEXR
            import Imath

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            height, width = depth.shape[-2:]
            depth_float = depth.astype(np.float32)

            # Create EXR header
            header = OpenEXR.Header(width, height)
            header['channels'] = {
                'Z': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }

            # Write file
            exr = OpenEXR.OutputFile(filepath, header)
            exr.writePixels({'Z': depth_float.tobytes()})
            exr.close()

            print(f"✓ Saved depth map (EXR): {filepath}")
            return filepath

        except ImportError:
            print("Warning: OpenEXR not available, saving as NPY instead")
            return ExportUtils.save_depth_npy(filepath.replace('.exr', '.npy'), depth)

    @staticmethod
    def save_depth_png16(filepath: str, depth: np.ndarray, scale_factor: float = 1000.0) -> str:
        """
        Save depth map as 16-bit PNG.

        Args:
            filepath: Output path (.png)
            depth: Depth array
            scale_factor: Scaling factor for quantization (default: 1000 for mm)
        """
        from PIL import Image

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Scale and convert to 16-bit
        depth_scaled = (depth * scale_factor).clip(0, 65535).astype(np.uint16)

        # Save as PNG
        img = Image.fromarray(depth_scaled, mode='I;16')
        img.save(filepath)

        print(f"✓ Saved depth map (PNG16): {filepath}")
        return filepath

    # ============================================================================
    # Camera Parameters Export
    # ============================================================================

    @staticmethod
    def save_camera_parameters_json(
        filepath: str,
        poses: np.ndarray,
        intrinsics: np.ndarray
    ) -> str:
        """
        Save camera parameters to JSON file.

        Args:
            filepath: Output path (.json)
            poses: Camera-to-world matrices [N, 4, 4]
            intrinsics: Intrinsic matrices [N, 3, 3]

        Returns:
            filepath: Path to saved file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        cameras = []
        for i in range(len(poses)):
            camera = {
                'frame_id': i,
                'pose': poses[i].tolist(),
            }

            # Only add intrinsics if available
            if intrinsics is not None:
                camera['intrinsics'] = intrinsics[i].tolist()
                camera['focal_length'] = [
                    float(intrinsics[i, 0, 0]),
                    float(intrinsics[i, 1, 1])
                ]
                camera['principal_point'] = [
                    float(intrinsics[i, 0, 2]),
                    float(intrinsics[i, 1, 2])
                ]

            cameras.append(camera)

        data = {
            'num_frames': len(poses),
            'coordinate_system': 'opencv',
            'pose_convention': 'camera_to_world',
            'cameras': cameras,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved camera parameters: {filepath} ({len(poses)} frames)")
        return filepath

    @staticmethod
    def save_camera_parameters_npy(
        filepath: str,
        poses: np.ndarray,
        intrinsics: np.ndarray
    ) -> str:
        """
        Save camera parameters as NumPy arrays.

        Args:
            filepath: Base output path (without extension)
            poses: Camera poses [N, 4, 4]
            intrinsics: Intrinsics [N, 3, 3]

        Returns:
            filepath: Base path used
        """
        base_path = Path(filepath).with_suffix('')
        base_path.parent.mkdir(parents=True, exist_ok=True)

        poses_path = f"{base_path}_poses.npy"
        intrinsics_path = f"{base_path}_intrinsics.npy"

        np.save(poses_path, poses.astype(np.float32))
        np.save(intrinsics_path, intrinsics.astype(np.float32))

        print(f"✓ Saved camera parameters: {poses_path}, {intrinsics_path}")
        return str(base_path)


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert tensor to numpy array.

    Args:
        tensor: PyTorch tensor or NumPy array

    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def save_depth_visualization(
    filepath: str,
    depth: np.ndarray,
    colormap: str = 'viridis',
    normalize: bool = True
) -> str:
    """
    Save depth map as colorized visualization.

    Args:
        filepath: Output path (.png or .jpg)
        depth: Depth array
        colormap: Matplotlib colormap name
        normalize: Normalize depth to [0, 1]

    Returns:
        filepath: Path to saved file
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Normalize if requested
    if normalize:
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth)

    # Convert to RGB and save
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    from PIL import Image
    img = Image.fromarray(colored_rgb)
    img.save(filepath)

    print(f"✓ Saved depth visualization: {filepath}")
    return filepath
