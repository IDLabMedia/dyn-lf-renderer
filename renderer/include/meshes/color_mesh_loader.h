/*
* Created by Brent Matthys on 25/04/2025
*/

#pragma once

#include "input/buffered_loader.h"
#include "meshes/mesh_loader.h"
#include <cstddef>
#include <string>

class ColorMesh: public MeshLoader{
protected:
  std::string _inDir;
  GLuint _colorVBO = 0;

  BufferedLoader<glm::uint8_t> _inpaintingBuffer;
  size_t _bufferedFrames;

  std::string colorPath(unsigned int frame) const;

public:
  ColorMesh(unsigned int width, unsigned int height, std::string inDir, size_t totalFrames, size_t bufferedFrames = 5);

  void fillVBO(unsigned int frame) override;

  void fillColorVBO(unsigned int frame);
  void fillColorVBO(const void* data, const size_t size);
};
