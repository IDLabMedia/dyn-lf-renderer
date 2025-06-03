#include "input/reader.h"
#include <string>


#define STB_IMAGE_IMPLEMENTATION // SHOULD ONLY BE DEFINED ONCE
#include "stb_image.h"

glm::uint8* readJPEGToBuffer(const std::string& filePath, glm::uint32& bufferSize) {
  int width, height, channels;
    
  // Load image using stb_image
  glm::uint8* data = stbi_load(filePath.c_str(), &width, &height, &channels, 0);
  if (!data) {
    std::cerr << "ERROR::STB::LOADING: Failed to load image: " << filePath << std::endl;
    exit(EXIT_FAILURE);
  }
  
  // output
  bufferSize = width * height * channels;
  return data;
}
