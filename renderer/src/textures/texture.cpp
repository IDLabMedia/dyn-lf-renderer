/**
* Created by Brent Matthys on 12/04/2025
*/

#include "textures/texture.h"
#include <utility>
#include <vector>

Texture::Texture(
    std::string textureName,
    bool linear,
    GLint internalFormat,
    GLenum format,
    GLenum type,
    GLuint width,
    GLuint height,
    GLuint depth
): 
  _textureName(std::move(textureName)),
  _linear(linear),
  _internalFormat(internalFormat),
  _format(format),
  _type(type),
  _width(width),
  _height(height),
  _depth(depth)
{
  allocate();
}

void Texture::assignToShader(const ShaderProgram& shaderProgram){
  _textureUnit = TextureUnitProvider::getUnit();
  shaderProgram.setInt(_textureName, _textureUnit);
}

void Texture::allocate(GLuint width, GLuint height, GLuint depth){
  // generate texture
  _textureId = 0;
  glGenTextures((GLsizei)1, &_textureId);

  // specify texture we are updating for
  glBindTexture(GL_TEXTURE_2D_ARRAY, _textureId);

  // set texture parameters
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, _linear ? GL_LINEAR : GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, _linear ? GL_LINEAR : GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // allocate memory for texture
  glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, _internalFormat, _width, _height, _depth, 0, _format, _type, nullptr);
}

void Texture::updateTextureLayers(const void* data, GLuint zOffset, GLuint layers){
  // specify texture we are updating for
  glBindTexture(GL_TEXTURE_2D_ARRAY, _textureId);
  // update texture
  glTexSubImage3D(
    GL_TEXTURE_2D_ARRAY, 0, 0, 0, zOffset, _width, _height, layers, _format, _type, data
  );
}

void Texture::updateTextureLayers(const std::vector<const void*>& data){
  for(GLuint layer = 0; layer < data.size(); ++layer){
    updateTextureLayers(data[layer], layer);
  }
}

GLuint Texture::getTextureId() const {
    return _textureId;
}

void Texture::bind() const {
    glActiveTexture(GL_TEXTURE0 + _textureUnit);
    glBindTexture(GL_TEXTURE_2D_ARRAY, _textureId);
}

const std::string& Texture::getTextureName() const {
    return _textureName;
}
