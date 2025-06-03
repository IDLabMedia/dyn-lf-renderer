#version 460 core
layout (location = 0) in vec3 vertex;

// uniforms
uniform vec2 resolution;        // The camera resolution
uniform vec2 pp;                // The principal point
uniform vec2 focal;             // The camera's focal
uniform vec2 depth_range;       // The near and far plane
uniform mat4 view;              // The view matrix (contains the view position on col 3)

out vec3 viewPos; // vertex position in camera space 

void main() {
    vec4 worldPos = vec4(vertex, 1.0);
    vec4 viewPosition = view * worldPos;

    viewPos = viewPosition.xyz; // pass full view-space position

    float u = viewPosition.x / viewPosition.z * focal.x + pp.x - 0.5;
    float v = viewPosition.y / viewPosition.z * focal.y + pp.y - 0.5;

    // map from pixel to NDC space
    float x_ndc = 2.0 * u / resolution.x - 1.0;
    float y_ndc = 2.0 * v / resolution.y - 1.0;

    gl_Position = vec4(x_ndc, y_ndc, 0.0, 1.0); // z is arbitrary, override it in fragment
}

