#version 460 core

layout (location = 0) out vec4 FragColor; // color of the pixel

in vec4 worldPos;

uniform sampler2DArray intrinsicsTexture; // the intrinsics for the cameras
uniform sampler2DArray invModelTexture; // the inverse models for the cameras
uniform sampler2DArray pngTexture; // the png images


// Function to retrieve a mat2 from the intrinsics texture
mat2 getIntrinsics(int cameraIdx) {
    vec4 row1 = texelFetch(intrinsicsTexture, ivec3(0, 0, cameraIdx), 0);
    vec4 row2 = texelFetch(intrinsicsTexture, ivec3(1, 0, cameraIdx), 0);

    return mat2(row1.xy, row2.xy);
}

// Function to retrieve a mat4 from the inverse model texture
mat4 getInvModel(int cameraIdx) {
    vec4 row1 = texelFetch(invModelTexture, ivec3(0, 0, cameraIdx), 0);
    vec4 row2 = texelFetch(invModelTexture, ivec3(1, 0, cameraIdx), 0);
    vec4 row3 = texelFetch(invModelTexture, ivec3(2, 0, cameraIdx), 0);
    vec4 row4 = texelFetch(invModelTexture, ivec3(3, 0, cameraIdx), 0);

    return mat4(row1, row2, row3, row4);
}

vec2 computeUV(int cameraIndex){
    mat2 intrinsics = getIntrinsics(cameraIndex); // get the intrinsics
    vec4 viewPos = getInvModel(cameraIndex) * worldPos; // to camera view space
    viewPos = viewPos / viewPos.w; // normalize

    float u = viewPos.x / viewPos.z * intrinsics[0][0] + intrinsics[1][0] - 0.5;
    float v = viewPos.y / viewPos.z * intrinsics[0][1] + intrinsics[1][1] - 0.5;

    u = u / 1920;
    v = v / 1080;

    return vec2(u,v);
}

void main(){

    int cameraIndex = 0;

    vec2 uv = computeUV(cameraIndex);
    FragColor = texture(pngTexture, vec3(uv, cameraIndex));
    FragColor = vec4(0.5, 0.0, 0.0, 1.0);

}
