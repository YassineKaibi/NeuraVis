#version 460 core

// Input: vertex position and weight value
layout(location = 0) in vec3 a_position;   // Vertex position (start or end)
layout(location = 1) in float a_weight;    // Weight value for this connection

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;

// Output to fragment shader
out float v_weight;

void main() {
    v_weight = a_weight;
    gl_Position = u_projection * u_view * vec4(a_position, 1.0);
}
