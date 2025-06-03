/**
* Created by Brent Matthys on 30/03/2025
*/

#pragma once

#include "codec/dct_matrix_provider.h"
#include "codec/structure/frame.h"
#include "codec/encoder/frame_transformer.h"

#include "threadpool.h"

class DCT: public FrameTransformer {
private:
  ThreadPool& _pool = ThreadPool::getInstance();

  DCTMatrixProvider _matrixProvider = DCTMatrixProvider();

protected:

  virtual void transform(const Block& inBlock, Block& outBlock, Channel channel) override;
};

