//
// Created by brent on 04/02/24.
//


#include "textures/jpeg_texture.h"

#include "fwd.hpp"
#include "input/reader.h"
#include <utility>

JPEGTexture::JPEGTexture(
  std::string inDir,
  const std::vector<glm::vec2> &resolutions,
  unsigned int totalFrames
):
  _inDir(std::move(inDir)),
  _textureBuffers(totalFrames, readJPEGToBuffer)
{
  _textures.emplace_back(
    "pngTexture",
    false,
    GL_RGB8,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    resolutions.front().x,
    resolutions.front().y,
    resolutions.size()
  );
  cameras = resolutions.size();
}


std::string JPEGTexture::computePath(unsigned int frame) const{
  return _inDir + "/frame_" + std::to_string(frame) + "/colors.jpeg";
}

void JPEGTexture::loadFrame(const unsigned int frame) {
  // load data from disk to GPU (or used cached data if available)
  _textures.back().updateTextureLayers(_textureBuffers.getBuffer(frame, computePath(frame)), 0, cameras);
  // delete the used buffer
  _textureBuffers.clearBuffer(frame);
  // ensure that 10 buffers ahead are loaded
  _textureBuffers.loadBuffers(frame + 1, 1, [this](const unsigned int newFrame) {return computePath(newFrame);});
  _textureBuffers.printLoadedBuffers();
}
