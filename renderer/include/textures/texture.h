/**
* Created by Brent Matthys on 12/04/2025
*/

#include "shaders/shader_program.h"
#include <glad/glad.h>
#include <string>
#include <vector>

class TextureUnitProvider {
public:
  /**
  * Generate incremental unique texture unit id's.
  */
  static GLuint getUnit(){
    static int currentUnit = 0;
    return currentUnit++;
  }
};

class Texture {
private:
  GLuint _textureId;

  GLuint _textureUnit;
  const std::string _textureName;

  const bool _linear;
  const GLint _internalFormat;
  const GLenum _format;
  const GLenum _type;

  const GLuint _width; // x dim
  const GLuint _height; // y dim
  const GLuint _depth; // z dim

public:

  Texture(
    std::string textureName,
    bool linear,
    GLint internalFormat,
    GLenum format,
    GLenum type,
    GLuint width,
    GLuint height,
    GLuint depth
  );

  /*
  * Bind this texture to a shaderProgram.
  * Only required to call this once per shaderProgram.
  */
  void assignToShader(const ShaderProgram& shaderProgram);

  /*
  * Allocate a 3D texture (2D_ARRAY) on the gpu.
  * width represents the x dimension.
  * height represents the y dimension.
  * depth represents the z dimension.
  */
  void allocate(GLuint width = 0, GLuint height = 0, GLuint depth = 0);

  /*
  * Update layers of the 2D_ARRAY.
  * data size should be _width*_height*layers
  */
  void updateTextureLayers(
    const void* data, GLuint zOffset = 0, GLuint layers = 1
  );

  /*
  * Update texture layers in the texture array.
  */
  void updateTextureLayers(const std::vector<const void*>& data);

  /**
   * Getter for the texture Id
   * @return The texture Id
   */
  GLuint getTextureId() const;

  /**
   * Bind this texture to the unit for this texture.
   * Only one texture should be bound per unit.
   */
  void bind() const;

  const std::string& getTextureName() const;
};


