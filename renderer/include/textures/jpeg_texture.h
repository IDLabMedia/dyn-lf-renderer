//
// Created by brent on 04/02/24.
//

#ifndef JPEG_TEXTURE_H
#define JPEG_TEXTURE_H

#include "fwd.hpp"
#include "input/buffered_loader.h"
#include "textures/textures_loader.h"
#include <string>

class JPEGTexture: public TexturesLoader {
private:
  std::string _inDir;
  BufferedLoader<glm::uint8> _textureBuffers;
  GLuint cameras;

  std::string computePath(unsigned int frame) const;

public:
  JPEGTexture(
    std::string inDir,
    const std::vector<glm::vec2>& resolutions,
    unsigned int totalFrames
  );

  void loadFrame(unsigned int frame) override;
};

#endif // !JPEG_TEXTURE_H
