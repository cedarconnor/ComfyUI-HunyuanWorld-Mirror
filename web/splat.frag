// Gaussian Splat Fragment Shader
// Evaluates Gaussian falloff and lets Three.js handle color space conversion.

precision highp float;

varying vec4 vColor;
varying vec2 vPosition;

void main() {
    // Compute Gaussian weight based on distance from center
    // vPosition is in [-1, 1] range for the quad
    float dist2 = dot(vPosition, vPosition);

    // Gaussian function: exp(-0.5 * dist^2)
    // Discard fragments beyond 3 standard deviations (dist = 1 in our normalized space)
    if (dist2 > 1.0) {
        discard;
    }

    float alpha = exp(-0.5 * dist2) * vColor.a;

    // Early discard for nearly transparent fragments
    if (alpha < 0.004) {
        discard;
    }

    // Clamp to avoid NaNs while keeping colors in linear space
    vec3 color = clamp(vColor.rgb, 0.0, 1.0);

    // Output color with computed alpha (Three.js handles gamma corrections)
    gl_FragColor = vec4(color * alpha, alpha);
}
