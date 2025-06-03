//
// Created by brent on 12/11/24.
//

#include "textures/raw_imgs_texture.h"

#include "fwd.hpp"
#include "input/reader.h"

RawImgsTexture::RawImgsTexture(
  std::string inDir,
  const std::vector<glm::vec2> &resolutions,
  unsigned int totalFrames
):
  _inDir(std::move(inDir)), _textureBuffers(totalFrames, readRawFileToBuffer<glm::uint8>)
{
  _textures.emplace_back(
    "rawImgs",
    false,
    GL_RGB8,
    GL_BGR,
    GL_UNSIGNED_BYTE,
    resolutions.front().x,
    resolutions.front().y,
    resolutions.size()
  ); 
  cameras = resolutions.size();
}

void RawImgsTexture::loadFrame(const unsigned int frame) {
    // load data from disk to GPU (or used cached data if available)
    _textures.back().updateTextureLayers(_textureBuffers.getBuffer(frame, computePath(frame)), 0, cameras);
    // delete the used buffer
    _textureBuffers.clearBuffer(frame);
    // ensure that 10 buffers ahead are loaded
    _textureBuffers.loadBuffers(frame + 1, 1, [this](const unsigned int newFrame) {return computePath(newFrame);});
}

std::string RawImgsTexture::computePath(const unsigned int frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/colors.raw";
}


