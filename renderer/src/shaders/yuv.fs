#version 460 core
out vec4 FragColor;

in vec4 worldPos; // vertex position in world space
in vec3 fallbackColor; // average color for this fragment to be used if no camra available for this fragment in yuv

// uniforms
uniform mat4 view;              // The view matrix (contains the view position on col 3)
uniform int cameras;            // The total amount of cameras
uniform vec2 resolutions[32];   // The resolution per camera
uniform vec2 pps[32];           // The principal points for each camera
uniform vec2 focals[32];        // The focals of each camera
uniform vec2 depth_range[32];   // The focals of each camera
uniform mat4 invMatrices[32];   // The inverse model matrices of each camera
uniform vec3 positions[32];  // The position of each camera

uniform int selectedCameraCount;// The amount of cameras in use for this frame
uniform int selectedCameras[32];    // The indices of the selected cameras

// textures (only selectedCameraCount textures available)
uniform sampler2DArray yTexture;
uniform sampler2DArray uTexture;
uniform sampler2DArray vTexture;
uniform sampler2DArray depthTexture;


// constants
vec3 vertexPos = worldPos.xyz;
vec3 viewPos = view[3].xyz;
float PI = 3.141592653589793f;

// functions

//
// For a given camera compute the uv texture coordinate
//
vec2 computeUV(int cameraIdx){
    vec2 resolution = resolutions[cameraIdx];
    vec2 pp = pps[cameraIdx];
    vec2 focal = focals[cameraIdx];

    // transform vertex from world space to camera space
    vec4 camSpacePos = invMatrices[cameraIdx] * worldPos;
    camSpacePos = camSpacePos / camSpacePos.w;

    // compute uv coordinate
    float u = camSpacePos.x / camSpacePos.z * focal.x + pp.x - 0.5; // u: col of the texture
    float v = camSpacePos.y / camSpacePos.z * focal.y + pp.y - 0.5; // v: row of the texture

    u = u / resolution.x;
    v = v / resolution.y;

    return vec2(u,v);
}

//
// Compute the angle of cvo. With c the camera position, v the vertex position and o the viewpoint position.
//
float computeAngle(int cameraIdx){
    vec3 camPos = positions[cameraIdx];

    // Compute the direction vectors
    vec3 VC = camPos - vertexPos;
    vec3 VO = viewPos - vertexPos;

    // Normalize the direction vectors
    vec3 VC_norm = normalize(VC);
    vec3 VO_norm = normalize(VO);

    // Compute the cosine of the angle
    float cosTheta = dot(VC_norm, VO_norm);

    // Clamp the value to avoid numerical issues
    cosTheta = clamp(cosTheta, -1.0, 1.0);

    // Compute and return the angle in radians
    return acos(cosTheta);
}

//
// Get the closest depth on the ray from this fragment to the camera
// in range [0,1], where 0 is on near and 1 is on far plane
//
float getClosestDepth(int cameraIdx, int textureIdx, vec2 uv){
  return texture(depthTexture, vec3(uv, textureIdx)).r;
}

//
//Get the depth from this fragment to the camera
//in range [0,1], where 0 is on near and 1 is on far plane
//
float getDepth(int cameraIdx){
  // transform to viewspace
  vec4 viewPosition = invMatrices[cameraIdx] * worldPos;
  viewPosition = viewPosition / viewPosition.w;

  float z = length(viewPosition.xyz); // get dist

  vec2 dr = depth_range[cameraIdx];
  float depth = (z - dr.x) / (dr.y - dr.x); // normalized depth
  //depth = (1/z - 1/dr.x) / (1/dr.y - 1/dr.x); // normalized depth

  return clamp(depth, 0.0, 1.0); // match what was written to depth buffer
}

vec4 yuvToRgb(float y, float u, float v) {
    // https://en.wikipedia.org/wiki/SYCC
    y = clamp(y, 0.0, 1.0);
    u = u - 0.5;
    v = v - 0.5;
    return vec4(
        y + 1.402 * v,
        y - 0.344136 * u - 0.714136 * v,
        y + 1.772 * u,
        1.0
    );
}

vec4 getColor(int textureIdx, vec2 uv){
  float y = texture(yTexture, vec3(uv, textureIdx)).r;
  float u = texture(uTexture, vec3(uv, textureIdx)).r;
  float v = texture(vTexture, vec3(uv, textureIdx)).r;

  return yuvToRgb(y,u,v);
}

//
// Return true if this point is occluded, false otherwise
//
bool visible(int cameraIdx, int textureIdx, vec2 uv) {
    float epsilon = 0.001;

    // Is this point visible to the camera (not occluded)?
    return  getDepth(cameraIdx) - epsilon <= getClosestDepth(cameraIdx, textureIdx, uv);
}

bool inTexRange(float x){
    return 0 <= x && x <= 1;
}

bool inTexRange(vec2 uv){
    return inTexRange(uv.x) && inTexRange(uv.y);
}


vec4 lightFieldRendering() {
    vec4 resultFragment = vec4(0.0); // Initialize resulting fragment
    float totalWeight = 0.0;        // Total weight accumulator for normalization
    bool cameraUsed = false;

    for (int textureIdx = 0; textureIdx < selectedCameraCount; textureIdx++) { // Iterate over all cameras
        int cameraIdx = selectedCameras[textureIdx];
        // Compute the UV texture coordinate for the current camera
        vec2 uv = computeUV(cameraIdx);
        // Check if the UV coordinate is within the valid range
        if (!inTexRange(uv)) continue;

        if(!visible(cameraIdx, textureIdx, uv)) continue;
        cameraUsed = true;

        // Compute the angle for the current camera
        float angle = computeAngle(cameraIdx);

        // linear weighting system 0° = 1, 180° = 0
        // float weight = (2 - angle/PI)/2.0f;

        // exponential decay weighting system 0° = 1, 180° = 0, but with steep dropoff at start
        float weight = exp(-angle);
        totalWeight += weight;

        // Sample the get the color for this camera
        vec4 fragment = getColor(textureIdx, uv);

        // Accumulate the weighted fragment
        resultFragment += weight * fragment;
    }

    // Normalize the result fragment by the total weight
    if (totalWeight > 0.0) {
        resultFragment /= totalWeight;
    }

    if(cameraUsed){
      return resultFragment; // Final blended fragment color
    }

    //return vec4(1.0f, 0.0f, 0.0f, 1.0f);
    return yuvToRgb(fallbackColor.x, fallbackColor.y, fallbackColor.z);
}

void main(){
    FragColor = lightFieldRendering();
    return;
    float d = getDepth(0);
    FragColor = vec4(d,d,d,1);
}
