#version 460 core
out vec4 FragColor;

// vertex position in world space
in vec4 worldPos;
in vec3 fallbackColor;


vec4 yuvToRgb(float y, float u, float v) {
    // https://en.wikipedia.org/wiki/SYCC
    y = clamp(y, 0.0, 1.0);
    u = u - (128.0f/255.0f);
    v = v - (128.0f/255.0f);

    float r = y + 0.000037 * u + 1.401988 * v;
    float g =    y + -0.344113 * u + -0.714104 * v;
    float b =    y + 1.771978 * u + 0.000135 * v;

    return vec4(clamp(vec3(r, g, b), 0.0, 1.0), 1.0);
}

void main(){
  
  FragColor = yuvToRgb(fallbackColor.r, fallbackColor.g, fallbackColor.b);
}
