#version 460 core

// Input
layout(location = 0) in vec3 a_position;  // 3D position from VBO

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_neuronSize;

// Output to fragment shader
out float v_activation;  // Pass activation value to fragment shader
out vec3 v_position;     // Pass position for debugging/effects

// SSBO for reading activations
layout(std430, binding = 2) readonly buffer ActivationsBuffer {
    float activations[];
} activationsData;

void main() {
    // Read activation value for this neuron
    v_activation = activationsData.activations[gl_VertexID];

    // Pass position to fragment shader
    v_position = a_position;

    // Transform position to clip space
    gl_Position = u_projection * u_view * vec4(a_position, 1.0);

    // Set point size (could scale by activation magnitude)
    // For now, use constant size
    gl_PointSize = u_neuronSize * 50.0;  // Scale factor for screen space
}