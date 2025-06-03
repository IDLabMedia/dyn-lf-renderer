#version 460 core
out vec4 FragColor;

// vertex position in world space
in vec4 worldPos;

void main(){
    float val = abs(worldPos.z);
    FragColor = vec4(val, val, val, 1.0f);
}
