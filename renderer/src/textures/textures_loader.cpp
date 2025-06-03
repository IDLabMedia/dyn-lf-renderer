//
// Created by brent on 11/20/24.
//

#include <glad/glad.h>

#include "textures/textures_loader.h"
#include "shaders/shader_program.h"

void TexturesLoader::assignTexturesToShader(const ShaderProgram& shaderProgram){
  for(auto& texture: _textures){
    texture.assignToShader(shaderProgram);
  }
}

void TexturesLoader::bindTextures(){
  for(auto& texture: _textures){
    texture.bind();
  }
}

