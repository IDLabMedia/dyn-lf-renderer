/**
* Created by Brent Matthys on 30/03/2025
*/

#include "codec/decoder/decoder.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_dir> <output_path>\n";
    return -1;
  }

  std::string inDir = argv[1];
  std::string outPath = argv[2];

  Decoder decoder = Decoder(inDir);
  auto frames = decoder.decode(0, {0});

  frames.front().write(outPath);
}

