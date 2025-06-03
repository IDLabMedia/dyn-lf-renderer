#version 460 core
layout (location = 0) in vec3 vertex;

// pass world position of the vertex to fragment shader
out vec4 worldPos;

uniform mat4 view;

void main(){
    worldPos = vec4(vertex, 1.0);

    // TODO load near and far plane as uniform
    float normalized_depth = (vertex.z - 0.3) / (1.62 - 0.3);
    gl_Position = view * vec4(vertex.x, -vertex.y, normalized_depth, 1.0);
}
