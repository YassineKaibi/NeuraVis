#version 460 core

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

// Input from vertex shader (per vertex)
in float v_weight[];

// Output to fragment shader
out float g_weight;

// Uniforms
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_lineWidth;  // Line width in world space

void main() {
    // Get the two endpoints of the line
    vec4 p0 = gl_in[0].gl_Position;  // Start point (clip space)
    vec4 p1 = gl_in[1].gl_Position;  // End point (clip space)

    // Convert to normalized device coordinates for perpendicular calculation
    vec2 p0_ndc = p0.xy / p0.w;
    vec2 p1_ndc = p1.xy / p1.w;

    // Calculate line direction in screen space
    vec2 dir = normalize(p1_ndc - p0_ndc);

    // Calculate perpendicular direction (rotate 90 degrees)
    vec2 perp = vec2(-dir.y, dir.x);

    // Scale perpendicular by line width (in NDC space)
    // We use a small multiplier since NDC is in [-1, 1] range
    vec2 offset = perp * u_lineWidth * 0.01;

    // Expand the line into a quad (4 vertices forming 2 triangles)
    // Triangle strip order: bottom-left, top-left, bottom-right, top-right

    // Vertex 0: Start bottom
    g_weight = v_weight[0];
    gl_Position = vec4(p0_ndc - offset, p0.z / p0.w, 1.0) * p0.w;
    EmitVertex();

    // Vertex 1: Start top
    g_weight = v_weight[0];
    gl_Position = vec4(p0_ndc + offset, p0.z / p0.w, 1.0) * p0.w;
    EmitVertex();

    // Vertex 2: End bottom
    g_weight = v_weight[1];
    gl_Position = vec4(p1_ndc - offset, p1.z / p1.w, 1.0) * p1.w;
    EmitVertex();

    // Vertex 3: End top
    g_weight = v_weight[1];
    gl_Position = vec4(p1_ndc + offset, p1.z / p1.w, 1.0) * p1.w;
    EmitVertex();

    EndPrimitive();
}
