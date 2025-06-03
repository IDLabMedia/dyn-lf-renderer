#version 460 core
layout (location = 0) in vec3 aPos;

out vec4 worldPos;

uniform mat4 view;

void main(){
    worldPos = vec4(aPos, 1.0);
    gl_Position = view * vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
