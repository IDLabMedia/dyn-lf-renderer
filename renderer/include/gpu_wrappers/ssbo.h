//
// Created by brent on 08/03/25.
//

#ifndef SSBO_H
#define SSBO_H


#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <glad/glad.h>

template <typename T>
class SSBO{
private:
  GLuint _ssboId;
  GLuint _elementCount;

  GLuint bufferSize() const {
    return _elementCount * sizeof(T);
  }

public:
  SSBO(): _ssboId(0), _elementCount(0) {
    glGenBuffers(1, &_ssboId);
  }
  ~SSBO() {
    glDeleteBuffers(1, &_ssboId);
  }

  /**
   * Resize the ssbo
   * @param size The new amount of elements to be storable in the ssbo.
   */
  void resize(const GLuint size) {
    _elementCount = size;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboId); // bind buffer
    glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize(), nullptr, GL_DYNAMIC_COPY); // allocate space
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind buffer
  }

  /**
  * Bind the ssbo to an id, that can be referenced by
  * the compute shader.
  * The latest ssbo bind to a bindId will be the
  * buffer available in the shader.
  *
  * @param bindId The id to bind the ssbo to
  */
  void bind(const GLuint bindId) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindId, _ssboId);
  }

  /**
  * Place data in the buffer at a given offset
  *
  * @param data The data to place
  * @param size The amount of elements to move to the ssbo
  * @param offset The amount of elements to skip before copying the data
  */
  void updateData(const void* data, const GLuint size, const GLuint offset){
    if(size + offset > bufferSize()){
      std::cerr << "ERROR::SSBO::UPDATE::OUT_OF_BOUNDS" << std::endl;
      exit(EXIT_FAILURE);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboId); // bind buffer
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size * sizeof(T), data); // update data
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // unbind buffer
  }

  /**
  * Get the amount of elements in the ssbo
  */
  GLuint size() const {
    return _elementCount;
  }

  /**
   * Get the data of the ssbo
   * @return A cpu copy of the gpu data of the ssbo
   */
  std::vector<T> data() const {
    std::vector<T> data(_elementCount);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _ssboId);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bufferSize(), data.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return data;
  }

  GLuint id() const {
    return _ssboId;
  }

};

#endif // !SSBO_H
