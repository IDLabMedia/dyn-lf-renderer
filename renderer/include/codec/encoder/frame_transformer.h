/**
* Created by Brent Matthys on 30/03/2025
*/

#pragma once


#include "codec/structure/frame.h"
#include "threadpool.h"

class FrameTransformer{
protected:
  ThreadPool& _pool = ThreadPool::getInstance();

  /**
   * Transform a single block. This function will be called in parallel for all blocks.
   */
  virtual void transform(const Block& inBlock, Block& outBlock, Channel channel) = 0;
public:

  /**
  * Transforms this frame block by block in parallel.
  */
  virtual void transform(const Frame& inFrame, Frame& outFrame);
};
