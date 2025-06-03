/**
* Created by Brent Matthys on 09/04/2025
*/

#include "codec/io/frame_writer.h"
#include "codec/structure/bit_buffer.h"
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>

void writeFrameBin(
  std::string path,
  std::vector<BitBuffer> camFrames,
  uint8_t blockSize,
  uint16_t frameWidth,
  uint16_t frameHeight,
  uint16_t qStep
){
  uint16_t numCameras = camFrames.size();

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  // STEP 1: Header data //
  out.write(reinterpret_cast<const char*>(&numCameras), sizeof(numCameras));
  out.write(reinterpret_cast<const char*>(&blockSize), sizeof(blockSize));
  out.write(reinterpret_cast<const char*>(&frameWidth), sizeof(frameWidth));
  out.write(reinterpret_cast<const char*>(&frameHeight), sizeof(frameHeight));
  out.write(reinterpret_cast<const char*>(&qStep), sizeof(qStep));

  // STEP 2: Compute and write offset table //
  std::vector<uint32_t> offsets(numCameras);
  std::vector<uint32_t> bitSizes(numCameras);

  uint32_t currentOffset = sizeof(uint8_t) + 4 * sizeof(uint16_t) + numCameras * (sizeof(uint32_t) * 2);

  for (uint16_t i = 0; i < numCameras; ++i) {
    bitSizes[i] = camFrames[i].getBitCount();
    offsets[i] = currentOffset;

    const auto& bytes = camFrames[i].getBytes();
    currentOffset += bytes.size();
  }

  for (uint16_t i = 0; i < numCameras; ++i) {
    out.write(reinterpret_cast<const char*>(&offsets[i]), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&bitSizes[i]), sizeof(uint32_t));
  }

  // 3: Write actual data //
  for (uint16_t i = 0; i < numCameras; ++i) {
    const auto& bytes = camFrames[i].getBytes();
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }

  out.close();

}

void writeRawYUV(std::string path, const std::vector<Frame> camFrames){
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  for(const auto& cameraFrame: camFrames){
    cameraFrame.write(out);
    /*out.write(reinterpret_cast<const char *>(cameraFrame.getY().data()), cameraFrame.getY().size());*/
    /*out.write(reinterpret_cast<const char *>(cameraFrame.getU().data()), cameraFrame.getU().size());*/
    /*out.write(reinterpret_cast<const char *>(cameraFrame.getV().data()), cameraFrame.getV().size());*/
  }

  out.close();
}
