/**
* Created by Brent Matthys on 10/04/2025
*/

#include "codec/decoder/decoder.h"
#include "codec/decoder/i_exp_golomb.h"
#include "codec/decoder/i_dct.h"
#include "codec/decoder/i_quantiser.h"
#include "codec/structure/frame.h"

#include <cstddef>
#include <future>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include <iostream>

Decoder::Decoder(std::string inDir): _inDir(inDir){}



std::string Decoder::getFramePath(size_t frame){
  return _inDir + "/frame_" + std::to_string(frame) + "/colours.bin";
}

void Decoder::getCompressedFrames(
  size_t frame, // in
  const std::vector<size_t>& cameras, // in
  std::vector<std::vector<uint8_t>>& compressedFrames, // out
  std::vector<size_t>& bitSizes, // out
  uint8_t& blockSize, // out
  uint16_t& frameWidth, // out
  uint16_t& frameHeight, // out
  uint16_t& qStep // out
){

  std::ifstream in(getFramePath(frame), std::ios::binary);
  if (!in) {
    throw std::runtime_error("DECODER::FAILED_TO_OPEN: " + getFramePath(frame));
  }

  // read header data
  uint16_t numCameras;
  in.read(reinterpret_cast<char*>(&numCameras), sizeof(numCameras));
  in.read(reinterpret_cast<char*>(&blockSize), sizeof(blockSize));
  in.read(reinterpret_cast<char*>(&frameWidth), sizeof(frameWidth));
  in.read(reinterpret_cast<char*>(&frameHeight), sizeof(frameHeight));
  in.read(reinterpret_cast<char*>(&qStep), sizeof(qStep));

  //  bounds check
  for (size_t cam : cameras) {
    if (cam >= numCameras) {
      throw std::runtime_error("DECODER::INVALID_CAMERA: " + std::to_string(cam));
    }
  }

  // read offset and bit size tables
  std::vector<uint32_t> offsets(numCameras);
  bitSizes = std::vector<size_t>(numCameras);

  for (size_t i = 0; i < numCameras; ++i) {
    in.read(reinterpret_cast<char*>(&offsets[i]), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&bitSizes[i]), sizeof(uint32_t));
  }

  // load each requested frame
  for (size_t cam : cameras) {
    uint32_t offset = offsets[cam];
    uint32_t bitSize = bitSizes[cam];
    uint32_t byteSize = (bitSize + 7) / 8;

    in.seekg(offset, std::ios::beg); // get location of cameraFrame
    std::vector<uint8_t> data(byteSize);
    in.read(reinterpret_cast<char*>(data.data()), byteSize); // read cameraframe
    compressedFrames.push_back(std::move(data));
  }
}

Frame Decoder::decodeCameraFrame(
  std::vector<uint8_t> compressed,
  size_t bitSize,
  size_t blockSize,
  size_t frameWidth,
  size_t frameHeight,
  size_t qStep
){
  Frame camFrame = Frame(frameWidth, frameHeight, blockSize);
  IExpGolomb entropyDecoder = IExpGolomb();
  IQuantiser iQuantiser = IQuantiser(qStep);
  IDCT idct = IDCT();

  Channel channel = Channel::Y;
  size_t bpRow = 0; // block pixel row
  size_t bpCol = 0; // block pixel col
  size_t bRow = 0;
  size_t bCol = 0;

  Block block = camFrame.getBlock(bRow, bCol);

  int symbol;
  std::vector<std::future<void>> futures;
  for(size_t i = 0; i < bitSize; ++i){
    // if bit doest result in complete symbol, continue
    if(!entropyDecoder.decode((compressed[i/8] >> (7 - (i % 8))) & 1, symbol)) continue;

    block.setAt(bpRow, bpCol, symbol, channel); // store the symbol in the frame

    bpCol++;
    if(bpCol != block.getWidth(channel)) continue; // no new row

    // new row
    bpCol = 0;
    bpRow++;
    if(bpRow != block.getHeight(channel)) continue; // not last row

    // last row => process block in parallel
    futures.push_back(
      _pool.enqueueTask(
        [&iQuantiser, &idct, &camFrame, bRow, bCol, channel](){
          Block block = camFrame.getBlock(bRow, bCol);
          iQuantiser.iQuantize(block, channel);
          idct.idct(block, channel);
        }
      )
    );

    bpRow = 0;
    bCol++;
    if(bCol != camFrame.getBlockCols()) {
      block = camFrame.getBlock(bRow, bCol); // update block
      continue;
    }

    bCol = 0;
    bRow++;
    if(bRow != camFrame.getBlockRows()) {
      block = camFrame.getBlock(bRow, bCol);
      continue;
    }

    // next channel
    channel = channel == Channel::Y ? Channel::U : Channel::V;
    bRow = 0;
    block = camFrame.getBlock(bRow, bCol);
  }

  for(auto& f: futures) f.get();

  return camFrame;
}

std::vector<Frame> Decoder::decode(size_t frame, std::vector<size_t> cameras){
  // STEP 1: read camera frames //
  std::vector<std::vector<uint8_t>> compressedFrames;
  std::vector<size_t> bitSizes;
  uint8_t blockSize;
  uint16_t frameWidth;
  uint16_t frameHeight;
  uint16_t qStep;

  getCompressedFrames(
    frame,
    cameras,
    compressedFrames,
    bitSizes,
    blockSize,
    frameWidth,
    frameHeight,
    qStep
  );

  std::vector<std::future<Frame>> futures;

  for(size_t i = 0; i < compressedFrames.size(); ++i){
    futures.push_back(
      _pool.enqueueTask(
        [this, &compressedFrames, &bitSizes, i, blockSize, frameWidth, frameHeight, qStep](){
          return this->decodeCameraFrame(
            compressedFrames[i], bitSizes[i], blockSize, frameWidth, frameHeight, qStep
          );
        }
      )
    );
  }

  std::vector<Frame> frames;
  for(auto& f: futures){
    frames.push_back(std::move(f.get()));
  }

  return frames;
}
