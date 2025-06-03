/**
* Created by Brent Matthys on 27/04/2025
*/

#pragma once

#include "meshes/mesh_loader.h"
#include "shaders/shader_program.h"
#include "textures/textures_loader.h"
#include "uniforms/camera_uniforms.h"
#include <string>

class DepthShader: public ShaderProgram {
private:

  GLuint _depthMapFBO;
  CameraUniforms _cameraData; // intrinsics and extrinsics of the cameras
  Texture _depthTexture; // texture to write depth info to
  
  void setUniforms(int camera, int textureIdx);
  void bindTextureToFramebuffer(int textureIdx);
  void computeDepth(const std::unique_ptr<MeshLoader>& mesh);

public:

  DepthShader(const std::string& inDir, const std::string& depthTextureName);

  /**
  * Renders the scene from the viewpoints of the current selected cameras.
  * Saves the depth info to texture.
  */
  void run(const std::unique_ptr<MeshLoader>& mesh);

  Texture& getDepthTexture();
};
