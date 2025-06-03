#version 460 core
layout (location = 0) in vec3 vertex;
layout (location = 1) in vec3 color;

// pass world position of the vertex to fragment shader
out vec4 worldPos;
out vec3 fallbackColor;

// uniforms
uniform mat4 view;              // The view matrix (contains the view position on col 3)
uniform int cameras;            // The total amount of cameras
uniform vec2 resolutions[32];   // The resolution per camera
uniform vec2 pps[32];           // The principal points for each camera
uniform vec2 focals[32];        // The focals of each camera
uniform vec2 depth_range[32];   // The focals of each camera
uniform mat4 invMatrices[32];   // The inverse model matrices of each camera
uniform vec3 positions[32];  // The position of each camera


void main(){
  fallbackColor = color;

  worldPos = vec4(vertex, 1.0);
  worldPos = worldPos / worldPos.w;

  vec2 out_res = resolutions[0];
  vec2 out_focal = focals[0];
  vec2 out_pp = pps[0];
  vec2 out_nf = depth_range[0];
  
  // project onto the output image
	vec4 viewPosition = view * worldPos;
	viewPosition = viewPosition / viewPosition.w;

  float u = viewPosition.x / viewPosition.z * out_focal.x + out_pp.x - 0.5;
  float v = viewPosition.y / viewPosition.z * out_focal.y + out_pp.y - 0.5;

  float normalised_depth = (viewPosition.z - out_nf.x) / (out_nf.y*2.0f - out_nf.x);
  viewPosition = vec4(2.0f * u / out_res.x - 1.0f, 2.0f * v / out_res.y - 1.0f, normalised_depth, 1.0f);
  gl_Position = viewPosition * vec4(1,-1,1,1); // image was inversed

}
