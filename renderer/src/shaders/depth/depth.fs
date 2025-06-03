#version 460 core

in vec3 viewPos; // vertex position in camera space 

uniform vec2 depth_range;       // The near and far plane

void main() {
    float z = length(viewPos);
    float depth = (z - depth_range.x) / (depth_range.y - depth_range.x);
    //depth = (1/z - 1/depth_range.x) / (1/depth_range.y - 1/depth_range.x);
    gl_FragDepth = clamp(depth, 0.0, 1.0);
}
