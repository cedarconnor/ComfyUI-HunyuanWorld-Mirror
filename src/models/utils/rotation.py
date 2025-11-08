# Rotation utilities for quaternions and rotation matrices
# References:
#   https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py

import torch
import torch.nn.functional as F


def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Tensor of shape (..., 4) containing quaternions in wxyz format

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    # Normalize quaternion
    quaternions = F.normalize(quaternions, p=2, dim=-1)

    w, x, y, z = torch.unbind(quaternions, -1)

    # Compute rotation matrix components
    wx = w * x
    wy = w * y
    wz = w * z
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z

    # Build rotation matrix
    rotmat = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
    ], dim=-2)

    return rotmat


def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions.

    Args:
        rotmat: Rotation matrices of shape (..., 3, 3)

    Returns:
        Quaternions of shape (..., 4) in wxyz format
    """
    # Based on "Converting a Rotation Matrix to a Quaternion" by Mike Day
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

    batch_shape = rotmat.shape[:-2]
    rotmat = rotmat.reshape(-1, 3, 3)

    # Extract rotation matrix elements
    m00, m01, m02 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    m10, m11, m12 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    m20, m21, m22 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]

    # Compute trace
    trace = m00 + m11 + m22

    # Initialize quaternion tensor
    quat = torch.zeros(rotmat.shape[0], 4, dtype=rotmat.dtype, device=rotmat.device)

    # Case 1: trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * w
    quat[mask1, 0] = 0.25 * s
    quat[mask1, 1] = (m21[mask1] - m12[mask1]) / s
    quat[mask1, 2] = (m02[mask1] - m20[mask1]) / s
    quat[mask1, 3] = (m10[mask1] - m01[mask1]) / s

    # Case 2: m00 > m11 and m00 > m22
    mask2 = (~mask1) & (m00 > m11) & (m00 > m22)
    s = torch.sqrt(1.0 + m00[mask2] - m11[mask2] - m22[mask2]) * 2  # s = 4 * x
    quat[mask2, 0] = (m21[mask2] - m12[mask2]) / s
    quat[mask2, 1] = 0.25 * s
    quat[mask2, 2] = (m01[mask2] + m10[mask2]) / s
    quat[mask2, 3] = (m02[mask2] + m20[mask2]) / s

    # Case 3: m11 > m22
    mask3 = (~mask1) & (~mask2) & (m11 > m22)
    s = torch.sqrt(1.0 + m11[mask3] - m00[mask3] - m22[mask3]) * 2  # s = 4 * y
    quat[mask3, 0] = (m02[mask3] - m20[mask3]) / s
    quat[mask3, 1] = (m01[mask3] + m10[mask3]) / s
    quat[mask3, 2] = 0.25 * s
    quat[mask3, 3] = (m12[mask3] + m21[mask3]) / s

    # Case 4: m22 is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + m22[mask4] - m00[mask4] - m11[mask4]) * 2  # s = 4 * z
    quat[mask4, 0] = (m10[mask4] - m01[mask4]) / s
    quat[mask4, 1] = (m02[mask4] + m20[mask4]) / s
    quat[mask4, 2] = (m12[mask4] + m21[mask4]) / s
    quat[mask4, 3] = 0.25 * s

    # Reshape back to original batch shape
    quat = quat.reshape(*batch_shape, 4)

    # Normalize quaternion
    quat = F.normalize(quat, p=2, dim=-1)

    return quat


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.

    Args:
        q1, q2: Quaternions of shape (..., 4) in wxyz format

    Returns:
        Product quaternion of shape (..., 4)
    """
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate.

    Args:
        q: Quaternion of shape (..., 4) in wxyz format

    Returns:
        Conjugate quaternion of shape (..., 4)
    """
    w, x, y, z = torch.unbind(q, -1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion inverse.

    Args:
        q: Quaternion of shape (..., 4) in wxyz format

    Returns:
        Inverse quaternion of shape (..., 4)
    """
    # For unit quaternions, inverse = conjugate
    q_conj = quat_conjugate(q)
    q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
    return q_conj / q_norm_sq
