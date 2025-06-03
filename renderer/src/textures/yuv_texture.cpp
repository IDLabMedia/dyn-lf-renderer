/**
* Created by Brent Matthys on 13/04/2025
*/

#include "textures/yuv_texture.h"
#include "camera_selector.h"
#include "fwd.hpp"
#include "input/reader.h"
#include "shaders/shader_program.h"
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

YUVTexture::YUVTexture(std::string inDir, GLuint frameWidth, GLuint frameHeight, const size_t totalFrames, bool headless):
  _inDir(inDir), 
  _lumaSize(frameWidth * frameHeight),
  _chromaSize(_lumaSize / 4),
  _totalFrames(totalFrames),
  _usedCameras(CameraSelector::getInstance().getSelectedCount()),
  _headless(headless)
{
  _textures.emplace_back(
    "yTexture",
    false,
    GL_R8,
    GL_RED,
    GL_UNSIGNED_BYTE,
    frameWidth,
    frameHeight,
    _usedCameras
  );
  _textures.emplace_back(
    "uTexture",
    false,
    GL_R8,
    GL_RED,
    GL_UNSIGNED_BYTE,
    frameWidth / 2,
    frameHeight / 2,
    _usedCameras
  );
  _textures.emplace_back(
    "vTexture",
    false,
    GL_R8,
    GL_RED,
    GL_UNSIGNED_BYTE,
    frameWidth / 2,
    frameHeight / 2,
    _usedCameras
  );

  if(!headless){
    _bufferedTextures.resize(_totalFrames, std::vector<glm::uint8_t*>(_usedCameras, nullptr));
    _bufferedCameras.resize(_totalFrames);

    auto cameras = CameraSelector::getInstance().getCameras();
    for(size_t i = 0; i < _bufferSize && i < totalFrames; ++i){
      bufferFrame(0 + i, cameras);
    }
  }
}

std::string YUVTexture::getFramePath(unsigned int frame) const{
  return _inDir + "/frame_" + std::to_string(frame) + "/colors.yuv";
}

std::string YUVTexture::getDepthPath(unsigned int frame) const{
  return _inDir + "/frame_" + std::to_string(frame) + "/depth.raw";
}

GLuint YUVTexture::getCameraFrameSize() const{
  return _lumaSize + 2 * _chromaSize;
}

void YUVTexture::bufferFrame(size_t frame, const std::vector<int>& cameras){
  std::string framePath = getFramePath(frame);

  _bufferedTextures[frame].assign(_usedCameras, nullptr);
  _bufferedCameras[frame].assign(cameras.begin(), cameras.begin() + std::min(_usedCameras, cameras.size()));

  size_t texIndex = _bufferedTextures.size() - 1;

  for(size_t i = 0; i < _bufferedCameras[frame].size(); ++i){
    int cam = _bufferedCameras[frame].at(i);

    _pool.enqueueTask(
      [this, frame, i, framePath, cam](){
        glm::uint32 frameSize = getCameraFrameSize();
        _bufferedTextures[frame][i] = readBuffer<glm::uint8>(framePath, cam*frameSize, frameSize);
      }
    );
  }
}

void YUVTexture::loadSingleFrame(unsigned int frame){
  const std::vector<int> cameras = CameraSelector::getInstance().getCameras();
  std::string framePath = getFramePath(frame);

  for(size_t i = 0; i < _usedCameras; ++i){
    GLuint camera = cameras[i];
    glm::uint32 frameSize = getCameraFrameSize();
    glm::uint8* data = readBuffer<glm::uint8>(framePath, camera*frameSize, frameSize);
    _textures[0].updateTextureLayers(data, i);
    _textures[1].updateTextureLayers(data + _lumaSize, i);
    _textures[2].updateTextureLayers(data + _lumaSize + _chromaSize, i);
    delete[] data; 
  }
}

void YUVTexture::loadFrame(unsigned int frame) {
  if(_headless){
    loadSingleFrame(frame);
    return;
  }
  std::vector<bool> loaded(_bufferedTextures[frame].size(), false);
  int notloaded = loaded.size();
  while(notloaded > 0){
    for(size_t i = 0; i < loaded.size(); ++i){
      glm::uint8_t* data = _bufferedTextures[frame][i];
      if(!loaded[i] && data != nullptr){
        _textures[0].updateTextureLayers(data, i);
        _textures[1].updateTextureLayers(data + _lumaSize, i);
        _textures[2].updateTextureLayers(data + _lumaSize + _chromaSize, i);
        delete[] data;
        loaded[i] = true;
        notloaded--;
      }
    }
  }

  bufferFrame((frame + _bufferSize) % _totalFrames, CameraSelector::getInstance().getCameras());
  CameraSelector::getInstance().selectCameras(_bufferedCameras[frame]); // set the selected cameras
}
