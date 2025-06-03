/**
* Created by Brent Matthys on 13/04/2025
*/


#include "textures/textures_loader.h"
#include "threadpool.h"
#include <cstddef>
#include <string>
#include <vector>

class YUVTexture: public TexturesLoader {
private:

  std::string _inDir;
  GLuint _lumaSize;
  GLuint _chromaSize;
  size_t _totalFrames;
  size_t _usedCameras;

  bool _headless;

  ThreadPool& _pool = ThreadPool::getInstance();

  std::string getFramePath(unsigned int frame) const;
  std::string getDepthPath(unsigned int frame) const;
  GLuint getCameraFrameSize() const;

  size_t _bufferSize = 3;
  /*
  * Clears current frame, and loads up to _bufferSize next frames (if not already loaded)
  */
  void bufferFrame(size_t frame, const std::vector<int>& cameras);


  std::vector<std::vector<glm::uint8*>> _bufferedTextures; // textures in the future
  std::vector<std::vector<int>> _bufferedCameras; // what cameras are buffered in the future
  
  void loadSingleFrame(unsigned int frame);

public:

  YUVTexture(
    std::string inDir,
    GLuint frameWidth,
    GLuint frameHeight,
    const size_t totalFrames,
    bool headless = false
  );

  void loadFrame(unsigned int frame) override;
};


