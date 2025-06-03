/**
* Created by Brent Matthys on 12/04/2025
*/

#include "shaders/shader_program.h"
#include "textures/textures_loader.h"
#include "codec/decoder/decoder.h"
#include <string>

class CodecTexture: public TexturesLoader{
private:
  Decoder _decoder;

public:

  CodecTexture(
    std::string inDir,
    GLuint frameWidth,
    GLuint unframeHeight
  );

  void loadFrame(unsigned int frame) override;
};


