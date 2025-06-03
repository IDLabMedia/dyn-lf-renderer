//
// Created by brent on 11/20/24.
//

#ifndef TEXTURE_LOADER_H
#define TEXTURE_LOADER_H

#include "shaders/shader_program.h"
#include "textures/texture.h"

#include <vector>

class TexturesLoader {
protected:
  std::vector<Texture> _textures;

public:
  void assignTexturesToShader(const ShaderProgram& shaderProgram);

  void bindTextures();

  /**
   * Load the texture data for a given frame.
   * @param frame The frame to load into the texture
   */
  virtual void loadFrame(unsigned int frame) = 0;
};

#endif //TEXTURE_LOADER_H
