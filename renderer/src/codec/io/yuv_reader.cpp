/**
* Created by Brent Matthys on 29/03/2025
*/

#include "codec/io/yuv_reader.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

std::vector<int> readRawYUVFrame(
  const std::string& path,
  const size_t frame,
  const size_t width,
  const size_t height
) {
  size_t lumaSize = width*height;
  size_t chromaSize = (width/2)*(height/2);
  size_t frameSize = lumaSize + 2 * chromaSize;

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("YUVREADER: Failed to open YUV file " + path);
  }

  file.seekg(frame * frameSize, std::ios::beg); // move file pointer to start of frame

  std::vector<uint8_t> buffer(frameSize);
  file.read(reinterpret_cast<char*>(buffer.data()), frameSize); // read buffer

  if (file.gcount() != static_cast<std::streamsize>(frameSize)) {
    throw std::runtime_error("Failed to read full YUV frame");
  }

  std::vector<int> pixBuffer(frameSize); // transform uint8_t to full int
  std::transform(buffer.begin(), buffer.end(), pixBuffer.begin(),
               [](uint8_t val) { return static_cast<int>(val); });

  file.close();
  return pixBuffer;
}
