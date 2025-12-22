#version 460 core

// Input from vertex shader
in float v_activation;
in vec3 v_position;

// Output
out vec4 FragColor;

// Simple colormap: map activation value to color
vec3 activationToColor(float activation) {
    // Map activation to [0, 1] range
    // Assuming activations are roughly in [-1, 1] range
    float normalized = (activation + 1.0) * 0.5;
    normalized = clamp(normalized, 0.0, 1.0);

    // Simple blue (low) -> green (mid) -> red (high) colormap
    vec3 color;
    if (normalized < 0.5) {
        // Blue to Green
        float t = normalized * 2.0;
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), t);
    } else {
        // Green to Red
        float t = (normalized - 0.5) * 2.0;
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t);
    }

    return color;
}

void main() {
    // Make points circular (not square)
    vec2 coord = gl_PointCoord * 2.0 - 1.0;  // Map to [-1, 1]
    float dist = length(coord);
    if (dist > 1.0) {
        discard;  // Discard fragments outside circle
    }

    // Get color based on activation
    vec3 color = activationToColor(v_activation);

    // Add some shading to make it look 3D
    float shading = 1.0 - dist * 0.3;  // Darker at edges
    color *= shading;

    FragColor = vec4(color, 1.0);
}