
#include "codec/encoder/frame_transformer.h"

#include <future>

void FrameTransformer::transform(const Frame& inFrame, Frame& outFrame) {
  std::vector<std::future<void>> futures;

  for (size_t row = 0; row < inFrame.getBlockRows(); ++row) {
    for (size_t col = 0; col < inFrame.getBlockCols(); ++col) {
      for (Channel channel : {Channel::Y, Channel::U, Channel::V}) { // Y, U, V
        futures.push_back(
          _pool.enqueueTask(
            [this, &inFrame, &outFrame, channel, row, col]() {
              const Block inBlock = inFrame.getBlock(row, col);
              Block outBlock = outFrame.getBlock(row, col);
              this->transform(inBlock, outBlock, channel);
            }
          )
        );
      }
    }
  }

  // await all tasks to finish
  for(auto& f: futures) f.get();
}
