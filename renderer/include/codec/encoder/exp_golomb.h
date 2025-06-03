/**
* Created by Brent Matthys on 09/04/2025
*/

#pragma once


#include "codec/structure/bit_buffer.h"
#include "codec/structure/frame.h"

#include "threadpool.h"

#include <cstdint>

class ExpGolomb {
private:
  ThreadPool& _pool = ThreadPool::getInstance();

  uint32_t signedToUnsigned(int32_t value) const;


  BitBuffer codeBlock(const Block& block, Channel channel) const;

public:
  BitBuffer expGolomb(int value) const;
  BitBuffer codeFrame(const Frame& frame) const;
};
