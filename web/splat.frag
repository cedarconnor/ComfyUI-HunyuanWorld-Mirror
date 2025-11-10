// Gaussian Splat Fragment Shader
// Evaluates Gaussian falloff and applies alpha blending

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

    // Output color with computed alpha
    gl_FragColor = vec4(vColor.rgb * alpha, alpha);
}
