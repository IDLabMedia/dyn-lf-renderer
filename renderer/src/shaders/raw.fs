#version 460 core
out vec4 FragColor;

// vertex position in world space
in vec4 worldPos;

// uniforms
uniform mat4 view;              // The view matrix (contains the view position on col 3)
uniform int cameras;            // The amount of cameras used
uniform vec2 resolutions[32];   // The resolution per camera
uniform vec2 pps[32];           // The principal points for each camera
uniform vec2 focals[32];        // The focals of each camera
uniform mat4 invMatrices[32];   // The inverse model matrices of each camera
uniform vec3 camPositions[32];  // The position of each camera

uniform vec4 ourColor;

// texture
uniform sampler2DArray rawImgs;


// constants
vec3 vertexPos = worldPos.xyz;
vec3 viewPos = view[3].xyz;
float PI = 3.141592653589793f;

// functions
/*
* For a given camera compute the uv texture coordinate
*/
vec2 computeUV(int cameraIdx){
    vec2 resolution = resolutions[cameraIdx];
    vec2 pp = pps[cameraIdx];
    vec2 focal = focals[cameraIdx];

    // transform vertex from world space to camera space
    vec4 camSpacePos = invMatrices[cameraIdx] * worldPos;

    // compute uv coordinate
    float u = (camSpacePos.x / camSpacePos.z) * focal.x + pp.x - 0.5f; // u: col of the texture
    float v = (camSpacePos.y / camSpacePos.z) * focal.y + pp.y - 0.5f; // v: row of the texture

    u = u / resolution.x;
    v = v / resolution.y;

    return vec2(u,v);
}

/*
* Compute the angle of cvo. With c the camera position, v the vertex position and o the viewpoint position.
*/
float computeAngle(int cameraIdx){
    vec3 camPos = camPositions[cameraIdx];

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

bool inTexRange(float x){
    return 0 <= x && x <= 1;
}

bool inTexRange(vec2 uv){
    return inTexRange(uv.x) && inTexRange(uv.y);
}

vec4 lightFieldRendering() {
    vec4 resultFragment = vec4(0.0); // Initialize resulting fragment
    float totalWeight = 0.0;        // Total weight accumulator for normalization

    for (int i = 0; i < cameras; i++) { // Iterate over all cameras
        // Compute the UV texture coordinate for the current camera
        vec2 uv = computeUV(i);

        // Check if the UV coordinate is within the valid range
        if (inTexRange(uv)) {
            // Compute the angle for the current camera
            float angle = computeAngle(i);

            // linear weighting system 0° = 1, 180° = 0
            float weight = 1 - angle/PI;
            totalWeight += weight;

            // Sample the texture for the current camera
            vec4 fragment = texture(rawImgs, vec3(uv, i));

            // Accumulate the weighted fragment
            resultFragment += weight * fragment;
        }
    }

    // Normalize the result fragment by the total weight
    if (totalWeight > 0.0) {
        resultFragment /= totalWeight;
    }

    return resultFragment; // Final blended fragment color
}



void main(){
    FragColor = lightFieldRendering();
}
