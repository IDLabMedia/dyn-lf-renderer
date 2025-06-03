/**
* Created by Brent Matthys on 30/03/2025
*/

#include "codec/structure/frame.h"
#include "threadpool.h"
#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>

struct EncoderArgs {
  std::string inDir;
  std::string outDir;

  size_t width;
  size_t height;
  size_t fps;
  size_t cameras;
  size_t frames;

  size_t blockSize;
  size_t qStep;

  void print() const {
    std::cout << "inDir: " << inDir << std::endl;
    std::cout << "outDir: " << outDir << std::endl;

    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "fps: " << fps << std::endl;
    std::cout << "cameras: " << cameras << std::endl;
    std::cout << "frames: " << cameras << std::endl;
    
    std::cout << "blockSize: " << blockSize << std::endl;
    std::cout << "qStep: " << qStep << std::endl;
  }
};

class Encoder{
private:
  ThreadPool& _pool = ThreadPool::getInstance();

  const EncoderArgs _args;

  std::vector<Frame> _currentFrame;

  std::vector<Frame> _outFrame;

  std::string getVidPath(size_t camera);
  std::string getFrameOutPath(size_t frame);
  std::string getYUVOutPath(size_t frame);

  std::vector<Frame> readFrame(size_t frame);
  void readCurrentFrame(size_t frame);

  void clearOutFrame();

public:
  explicit Encoder(EncoderArgs args);

  void encode();
  void raw();
};

