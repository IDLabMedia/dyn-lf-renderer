/**
* Created by Brent Matthys on 27/04/2025
*/

#include "shaders/depth_shader.h"
#include "camera_selector.h"
#include "shaders/shader_program.h"
#include "textures/textures_loader.h"
#include <cstddef>
#include <string>

DepthShader::DepthShader(const std::string& inDir, const std::string& depthTextureName):
  ShaderProgram({
    ShaderInfo(glCreateShader(GL_VERTEX_SHADER), "depth/depth.vs"),
    ShaderInfo(glCreateShader(GL_FRAGMENT_SHADER), "depth/depth.fs")
  }), 
  _cameraData(inDir),
  _depthTexture(
    depthTextureName,
    false,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT,
    GL_FLOAT, 
    _cameraData.getResolutions().front().x,
    _cameraData.getResolutions().front().y,
    CameraSelector::getInstance().getSelectedCount()
  )
{
  glGenFramebuffers(1, &_depthMapFBO);
}

void DepthShader::setUniforms(int camera, int textureIdx){
    setVec2("resolution", _cameraData.getResolutions()[camera]);
    setVec2("pp", _cameraData.getPps()[camera]);
    setVec2("focal", _cameraData.getFocals()[camera]);
    setVec2("depth_range", _cameraData.getDepthRange()[camera]);
    setMat4("view", _cameraData.getInverseMatrices()[camera]);
}

void DepthShader::bindTextureToFramebuffer(int textureIdx){
  glBindFramebuffer(GL_FRAMEBUFFER, _depthMapFBO);
  glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, _depthTexture.getTextureId(), 0, textureIdx);
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);  
}

void DepthShader::computeDepth(const std::unique_ptr<MeshLoader>& mesh){

  glViewport(0,0,1920,1080);
  glBindFramebuffer(GL_FRAMEBUFFER, _depthMapFBO);
  glClear(GL_DEPTH_BUFFER_BIT);

  _depthTexture.bind();
  
  mesh->bindVAOAndDraw();

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void DepthShader::run(const std::unique_ptr<MeshLoader>& mesh){
  // use this shader
  this->use();

  auto cameras = CameraSelector::getInstance().getCameras();
  for(size_t textureIdx = 0; textureIdx < cameras.size(); ++textureIdx){
    auto camera = cameras[textureIdx];
    setUniforms(camera, textureIdx); // make uniforms available
    bindTextureToFramebuffer(textureIdx); // assign correct texture location to write to
    computeDepth(mesh);
  }
}

Texture& DepthShader::getDepthTexture(){
  return _depthTexture;
}

