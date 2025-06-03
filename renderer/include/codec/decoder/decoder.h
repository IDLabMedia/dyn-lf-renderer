/**
* Created by Brent Matthys on 10/04/2025
*/

#include "codec/structure/frame.h"
#include "threadpool.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Decoder {
private:

  ThreadPool& _pool = ThreadPool::getInstance();

  const std::string _inDir;

  std::string getFramePath(size_t frame);
  void getCompressedFrames(
    size_t frame,
    const std::vector<size_t>& cameras,
    std::vector<std::vector<uint8_t>>& compressedFrames,
    std::vector<size_t>& bitSizes,
    uint8_t& blockSize,
    uint16_t& frameWidth,
    uint16_t& frameHeight,
    uint16_t& qStep
  );

  Frame decodeCameraFrame(
    std::vector<uint8_t> compressed,
    size_t bitSize,
    size_t blockSize,
    size_t frameWidth,
    size_t frameHeight,
    size_t qStep
  );

public:

  explicit Decoder(std::string inDir);

  /**
  * Load a single frame from an encoded file and decode it.
  * Only extract and decode the frames of the requested cameras.
  */
  std::vector<Frame> decode(size_t frame, std::vector<size_t> cameras);
};


