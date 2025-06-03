/**
* Created by Brent Matthys on 09/04/2025
*/


#include "codec/encoder/exp_golomb.h"

#include "codec/structure/bit_buffer.h"
#include "codec/structure/frame.h"

#include <cstdint>
#include <future>

uint32_t ExpGolomb::signedToUnsigned(int32_t value) const{
  return value <= 0 ? (-2*value) : (2*value - 1);
}

BitBuffer ExpGolomb::expGolomb(int value) const{
  uint32_t codeNum = signedToUnsigned(value) + 1;
  uint32_t tmp = codeNum;
  uint32_t numBits = 0;

  while (tmp >>= 1) ++numBits; // count the amount of bits needed to represent tmp (minus one)

  BitBuffer code;
  code.pushBits(0, numBits); // prefix zeros equal to number bits used -1
  code.pushBits(codeNum, numBits + 1); // write codeNum
  return code;
}

BitBuffer ExpGolomb::codeBlock(const Block& block, Channel channel) const{
  BitBuffer code;
  for(size_t row = 0; row < block.getHeight(channel); ++row){
    for(size_t col = 0; col < block.getWidth(channel); ++col){
      int val = block.getAt(row, col, channel);
      code.pushBitBuffer(expGolomb(val));
    }
  }
  return code;
}

BitBuffer ExpGolomb::codeFrame(const Frame& frame) const{
  BitBuffer code;
  for(Channel channel : {Channel::Y, Channel::U, Channel::V}){
    std::vector<std::future<BitBuffer>> futures;
    // map blocks to encoded bit buffers
    for (size_t row = 0; row < frame.getBlockRows(); ++row) {
      for (size_t col = 0; col < frame.getBlockCols(); ++col) {
        futures.push_back(
          _pool.enqueueTask(
            [this, &frame, row, col, channel](){
              const Block block = frame.getBlock(row, col);
              return this->codeBlock(block, channel);
            }
          )
        );
      }
    }
    // reduce to single bit buffer
    for(auto& f: futures){
      code.pushBitBuffer(f.get());
    }
  }
  return code;
}
