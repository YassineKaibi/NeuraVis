#version 460 core

// Input from vertex shader
in float v_weight;

// Output
out vec4 FragColor;

// Map weight to color
// Positive weights = warm colors, negative = cool colors
vec3 weightToColor(float weight) {
    // Normalize weight to [-1, 1] range (adjust based on your network)
    float normalizedWeight = clamp(weight / 3.0, -1.0, 1.0);

    vec3 color;
    if (normalizedWeight < 0.0) {
        // Negative weights: blue
        float intensity = abs(normalizedWeight);
        color = mix(vec3(0.2, 0.2, 0.2), vec3(0.2, 0.4, 1.0), intensity);
    } else {
        // Positive weights: red/orange
        float intensity = normalizedWeight;
        color = mix(vec3(0.2, 0.2, 0.2), vec3(1.0, 0.4, 0.2), intensity);
    }

    return color;
}

void main() {
    vec3 color = weightToColor(v_weight);

    // Add transparency based on weight magnitude (weak connections more transparent)
    float alpha = mix(0.15, 0.8, clamp(abs(v_weight) / 3.0, 0.0, 1.0));

    FragColor = vec4(color, alpha);
}
