// Gaussian Splat Vertex Shader
// Based on 3D Gaussian Splatting formulation

// Per-instance attributes
attribute vec3 center;          // Gaussian center position
attribute vec3 scale;           // Gaussian scale (sx, sy, sz)
attribute vec4 rotation;        // Rotation quaternion (w, x, y, z)
attribute float opacity;        // Opacity value
attribute vec3 color;           // Base color (from f_dc_* or RGB)

// Per-vertex attributes (for quad corners)
attribute vec2 position;        // Corner position in [-1, 1] range

// Uniforms
uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;
uniform vec2 viewport;          // Viewport dimensions
uniform float scale_modifier;   // Additional scale factor

// Varyings to fragment shader
varying vec4 vColor;
varying vec2 vPosition;

// Quaternion to rotation matrix
mat3 quaternion_to_matrix(vec4 q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    return mat3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)
    );
}

void main() {
    // Transform center to view space
    vec4 viewCenter = modelViewMatrix * vec4(center, 1.0);

    // Create covariance matrix in 3D from rotation and scale
    mat3 R = quaternion_to_matrix(rotation);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 M = R * S;
    mat3 Sigma = M * transpose(M);

    // Project to 2D covariance (using Jacobian of perspective projection)
    float focal = viewport.x * projectionMatrix[0][0] / 2.0;
    float z = -viewCenter.z;
    float z2 = z * z;

    mat3 J = mat3(
        focal / z, 0.0, 0.0,
        0.0, focal / z, 0.0,
        -focal * viewCenter.x / z2, -focal * viewCenter.y / z2, 0.0
    );

    mat3 W = transpose(mat3(modelViewMatrix));
    mat3 T = W * J;
    mat3 cov2d = transpose(T) * Sigma * T;

    // Add a small value to diagonal for numerical stability
    float diagonal_offset = 0.1;
    cov2d[0][0] += diagonal_offset;
    cov2d[1][1] += diagonal_offset;

    // Compute eigenvalues to get the ellipse axes
    float mid = 0.5 * (cov2d[0][0] + cov2d[1][1]);
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);

    // Compute the 2D rotation angle
    float angle = atan(cov2d[0][1], lambda1 - cov2d[1][1]);
    mat2 rot2d = mat2(
        cos(angle), sin(angle),
        -sin(angle), cos(angle)
    );

    // Compute the quad extents (3 standard deviations)
    vec2 extent = 3.0 * sqrt(vec2(lambda1, lambda2)) * scale_modifier;

    // Apply rotation to corner position
    vec2 offset = rot2d * (position * extent);

    // Project center to clip space
    vec4 clipCenter = projectionMatrix * viewCenter;

    // Add offset in NDC space
    vec4 clipPos = clipCenter;
    clipPos.xy += offset / viewport * clipCenter.w;

    gl_Position = clipPos;

    // Pass data to fragment shader
    vColor = vec4(color, opacity);
    vPosition = position;
}
