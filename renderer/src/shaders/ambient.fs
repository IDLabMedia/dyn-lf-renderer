#version 460 core
out vec4 FragColor;

// vertex position in world space
in vec4 worldPos;
in vec3 fallbackColor;

void main(){
  
  FragColor = vec4(fallbackColor.b, fallbackColor.g, fallbackColor.r, 1.0f);
}
