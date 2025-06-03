/**
* Created by Brent Matthys on 30/03/2025
*/

#include "codec/encoder/encoder.h"

#include "codec/encoder/quantiser.h"
#include "codec/encoder/exp_golomb.h"

#include "codec/io/frame_writer.h"

#include "codec/structure/bit_buffer.h"
#include "codec/encoder/dct.h"

#include <chrono>
#include <cstddef>
#include <future>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

Encoder::Encoder(EncoderArgs args): _args(std::move(args)){}



std::string Encoder::getVidPath(size_t camera){
  camera += 1;
  std::ostringstream oss;
  oss << std::setw(2) << std::setfill('0') << camera;
  std::string camNumber = oss.str();

  return _args.inDir + "/v" + camNumber + "_texture.yuv";
}

std::string Encoder::getFrameOutPath(size_t frame){
  return _args.outDir + "/frame_" + std::to_string(frame) + "/colours.bin" ;

}

std::string Encoder::getYUVOutPath(size_t frame){
  return _args.outDir + "/frame_" + std::to_string(frame) + "/colors.yuv" ;
}

std::vector<Frame> Encoder::readFrame(size_t frame){
  std::vector<Frame> frames;
  std::vector<std::future<Frame>> futures;

  // run the read functions in parallel
  for(size_t cam = 0; cam < _args.cameras ; ++cam){
    futures.push_back(
      _pool.enqueueTask([this, cam, frame]() {
        return Frame(getVidPath(cam), frame, _args.width, _args.height, _args.blockSize);
      })
    );
  }
  
  // get the results
  for(auto& future: futures){
    frames.push_back(future.get());
  }
  return frames;
}

void Encoder:: readCurrentFrame(size_t frame){
  _currentFrame.clear();
  _currentFrame = readFrame(frame);
}

void Encoder::clearOutFrame(){
  // init empty output frame
  _outFrame.clear();
  for(size_t _ = 0; _ < _args.cameras; ++_){
    _outFrame.emplace_back(_args.width, _args.height, _args.blockSize);
  }
}

void Encoder::encode(){
  DCT dct = DCT();
  Quantiser quantiser = Quantiser(_args.qStep);
  ExpGolomb entropyCoder = ExpGolomb();

  auto start = std::chrono::steady_clock::now();

  // compress all the frames one by one
  for(size_t frame = 0; frame < _args.frames; ++frame){ // iterate each frame
    readCurrentFrame(frame); // read the input frame of each camera
    clearOutFrame(); // init the output frame
    std::vector<BitBuffer> frameCodes;
    for(size_t cam = 0; cam < _args.cameras; ++cam){ // iterate each camera
      
      // STEP 1: DCT //
      dct.FrameTransformer::transform(_currentFrame[cam], _outFrame[cam]); 

      // STEP 2: Quantization //
      quantiser.FrameTransformer::transform(_outFrame[cam], _outFrame[cam]);

      // STEP 3: Entropy coding //
      frameCodes.push_back(entropyCoder.codeFrame(_outFrame[cam]));
    }

    // STEP 4: Write frame to file //
    const Frame& f = _currentFrame.back();
    writeFrameBin(
      getFrameOutPath(frame),
      frameCodes,
      _args.blockSize,
      _args.width,
      _args.height,
      _args.qStep
    );

    // compute eta
    auto now = std::chrono::steady_clock::now();

    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(now - start).count();
    double avgTimePerFrame = elapsedSec / (frame + 1);
    size_t remainingFrames = _args.frames - frame - 1;
    double etaSec = avgTimePerFrame * remainingFrames;

    std::cout << "\r" << "Frame: " << frame << "/" << _args.frames << "; ETA: " << etaSec << "s" << std::flush;
  }
}


void Encoder::raw(){
  std::vector<std::future<void>> futures;
  for(size_t frame = 0; frame < _args.frames; ++frame){
    std::cout << "\r" << "Progress: " << frame << "/" << _args.frames << std::flush;
    writeRawYUV(this->getYUVOutPath(frame), readFrame(frame));
  }
}
