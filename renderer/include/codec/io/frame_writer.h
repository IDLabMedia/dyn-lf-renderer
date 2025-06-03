/**
* Created by Brent Matthys on 09/04/2025
*/

#include "codec/structure/bit_buffer.h"
#include "codec/structure/frame.h"
#include <cstdint>
#include <string>
#include <vector>

/**
* Write a single frame file.
* The file contains the number of number of cameras
* followed by a jump table to the frame data of each camera
* followed by the frames.
*
* [uint16_t numCameras]
* [uint8_t blockSize]
* [uint16_t frameWidth]
* [uint16_t frameHeight]
* [uint16_t qStep]
* [uint32_t offset_0][uint32_t bitsize_0]
* [uint32_t offset_1][uint32_t bitsize_1]
* ...
* [uint32_t offset_N-1][uint32_t bitsize_N-1]
* [data_0][data_1]...[data_N-1]
*
*/
void writeFrameBin(
  std::string path,
  std::vector<BitBuffer> camFrames,
  uint8_t blockSize,
  uint16_t frameWidth,
  uint16_t frameHeight,
  uint16_t qStep
);

void writeRawYUV(std::string path, const std::vector<Frame> camFrames);
