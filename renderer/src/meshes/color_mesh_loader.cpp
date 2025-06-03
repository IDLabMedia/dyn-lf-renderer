/*
* Created by Brent Matthys on 25/04/2025
*/

#include <cstddef>
#include <glad/glad.h>
#include <string>
#include <utility>
#include "meshes/color_mesh_loader.h"
#include "fwd.hpp"
#include "input/reader.h"
#include "meshes/mesh_loader.h"

ColorMesh::ColorMesh(unsigned int width, unsigned int height, std::string inDir, const size_t totalFrames, size_t bufferedFrames):
  MeshLoader(width, height),
  _inDir(std::move(inDir)),
  _inpaintingBuffer(totalFrames, readRawFileToBuffer<glm::uint8_t>),
  _bufferedFrames(bufferedFrames)
{
  glGenBuffers(1, &_colorVBO);
  _inpaintingBuffer.loadBuffers(0,_bufferedFrames, 
    [this](const unsigned int x){return this->colorPath(x);}
  );
}

std::string ColorMesh::colorPath(const unsigned int frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/fallback_colors.rgb";
}

void ColorMesh::fillVBO(unsigned int frame){
  // The VBO should be set separately
  // This only sets the colors
  fillColorVBO(frame);
}

void ColorMesh::fillColorVBO(unsigned int frame){
  glm::uint8_t* data = _inpaintingBuffer.getBuffer(frame, colorPath(frame));
  glm::uint32 size = _inpaintingBuffer.getBufferSize(frame);
  fillColorVBO(data, size);
  _inpaintingBuffer.clearBuffer(frame);
  _inpaintingBuffer.loadBuffers(frame + 1, _bufferedFrames, [this](const unsigned int x){return this->colorPath(x);});
}

void ColorMesh::fillColorVBO(const void* data, const size_t size){
  glBindVertexArray(_VAO);

  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glBufferData(GL_ARRAY_BUFFER, size * sizeof(glm::uint8_t), data, GL_STATIC_DRAW);

  glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 3*sizeof(glm::uint8_t), (void*)0);
  glEnableVertexAttribArray(1);
}
