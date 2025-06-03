/**
* Created by Brent Matthys on 30/03/2025
*/

#pragma once


#include "codec/encoder/frame_transformer.h"
#include <cstddef>

class Quantiser: public FrameTransformer{
private:
  size_t _qStep;
  
protected:
  virtual void transform(const Block& inBlock, Block& outBlock, Channel channel) override;
public:
  explicit Quantiser(size_t qStep);
};
