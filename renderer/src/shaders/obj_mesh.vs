// This vertex shader is called for each vertex in the EBO. It is designed to render an obj file mesh

#version 460 core
layout (location = 0) in vec3 vertex; // input from the VBO, i.e. (u,v) texture coordinates

out vec4 worldPos;

uniform mat4 view;

void main(){
    // project world position onto the output image
    vec4 viewPosition = view * vec4(vertex, 1.0);
    viewPosition = viewPosition / viewPosition.w;

    if(viewPosition.z < 0){
        float u = viewPosition.x / viewPosition.z * 1546.738242 + 980.168232;
        float v = viewPosition.y / viewPosition.z * 1547.755165 + 534.722224;
        float normalised_depth = (-viewPosition.z - 0.1f) / (3.24 - 0.1f);
        gl_Position = vec4(2.0f * u / 1920 - 1.0f, 2.0f * v / 1080 - 1.0f, normalised_depth, 1.0f);
    }
    else {
        // discard this vertex by placing it behind the near plane (it will be culled)
        gl_Position = vec4(0,0,-10,1);
    }

    viewPosition = vec4(viewPosition.x, -viewPosition.y, viewPosition.z, viewPosition.w);
    gl_Position = viewPosition;

    worldPos = vec4(vertex, 1.0); // pass the world position to the fragment shader
}
