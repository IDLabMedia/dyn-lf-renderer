/**
* Created by Brent Matthys on 30/03/2025
*/


#include <cstddef>
#include <iostream>
#include <ostream>
#include <unordered_map>
#include <string>
#include <utility>

#include "codec/encoder/encoder.h"

void printHelp(std::ostream& stream, const std::string& programName){
  stream << "Usage: " << programName << " <inDir> <outDir> [--width <int>] [--height <int>] [--fps <int>] [--blocksize <int>] [--qstep <int>]" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printHelp(std::cerr, argv[0]);
    return 1;
  }

  std::string inDir = argv[1];
  std::string outDir = argv[2];

  // Default values
  std::unordered_map<std::string, size_t> options = {
      {"--width", 1920},
      {"--height", 1080},
      {"--fps", 30},
      {"--blocksize", 16},
      {"--qstep", 5}
  };

  // Bool flags
  std::unordered_map<std::string, bool> flags = {
    {"--raw", false}
  };

  // Parse command-line arguments
  for (int i = 3; i < argc;) {
    std::string key = argv[i];
    if(flags.count(key)){
      flags[key] = true;
      ++i;
    } else if (i + 1 < argc) { // Ensure there is a value after the keyword arg
      int value = std::stoi(argv[i + 1]);
      if (options.find(key) != options.end()) {
        options[key] = value;
      }
      i += 2;
    }else{
      printHelp(std::cerr, argv[0]);
      return 1;
    }
  }

  // TODO dynamic values
  EncoderArgs args = {
    inDir, outDir, options["--width"], options["--height"], options["--fps"], 13, 300, options["--blocksize"], options["--qstep"]
  };

  Encoder encoder = Encoder(std::move(args));
  if(flags["--raw"]){
    encoder.raw();
  }else{
    encoder.encode();
  }

  return 0;
}
