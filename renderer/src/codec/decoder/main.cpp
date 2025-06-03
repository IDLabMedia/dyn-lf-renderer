/**
* Created by Brent Matthys on 30/03/2025
*/

#include "codec/decoder/decoder.h"

#include <string>

int main(int argc, char* argv[]) {
  std::string inDir = "/home/brent/projects/rtdlf/out";
  std::string outPath = "/home/brent/projects/rtdlf/decodes/frame_0_0.yuv";

  Decoder decoder = Decoder(inDir);
  auto frames = decoder.decode(0, {0});

  frames.front().write(outPath);
}

