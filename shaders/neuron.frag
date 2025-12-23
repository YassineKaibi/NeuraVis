#version 460 core

// Input from vertex shader
in float v_activation;
in vec3 v_position;

// Output
out vec4 FragColor;

// Viridis colormap: perceptually uniform, colorblind-friendly
// Based on matplotlib's viridis colormap
vec3 viridis(float t) {
    const vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    const vec3 c1 = vec3(0.282623, 0.140926, 0.457517);
    const vec3 c2 = vec3(0.253935, 0.265254, 0.529983);
    const vec3 c3 = vec3(0.206756, 0.371758, 0.553117);
    const vec3 c4 = vec3(0.163625, 0.471133, 0.558148);
    const vec3 c5 = vec3(0.127568, 0.566949, 0.550556);
    const vec3 c6 = vec3(0.134692, 0.658636, 0.517649);
    const vec3 c7 = vec3(0.266941, 0.748751, 0.440573);
    const vec3 c8 = vec3(0.477504, 0.821444, 0.318195);
    const vec3 c9 = vec3(0.741388, 0.873449, 0.149561);
    const vec3 c10 = vec3(0.993248, 0.906157, 0.143936);

    t = clamp(t, 0.0, 1.0);

    if (t < 0.1) return mix(c0, c1, t * 10.0);
    else if (t < 0.2) return mix(c1, c2, (t - 0.1) * 10.0);
    else if (t < 0.3) return mix(c2, c3, (t - 0.2) * 10.0);
    else if (t < 0.4) return mix(c3, c4, (t - 0.3) * 10.0);
    else if (t < 0.5) return mix(c4, c5, (t - 0.4) * 10.0);
    else if (t < 0.6) return mix(c5, c6, (t - 0.5) * 10.0);
    else if (t < 0.7) return mix(c6, c7, (t - 0.6) * 10.0);
    else if (t < 0.8) return mix(c7, c8, (t - 0.7) * 10.0);
    else if (t < 0.9) return mix(c8, c9, (t - 0.8) * 10.0);
    else return mix(c9, c10, (t - 0.9) * 10.0);
}

// Map activation value to color using viridis
vec3 activationToColor(float activation) {
    // Map activation to [0, 1] range
    // Assuming activations are roughly in [0, 2] range (ReLU outputs)
    // Adjust this range based on your network's activation function
    float normalized = clamp(activation / 2.0, 0.0, 1.0);

    return viridis(normalized);
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