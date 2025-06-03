/**
* Created by Brent Matthys on 30/03/2025
*/

#include "codec/encoder/quantiser.h"

#include <cmath>
#include <cstddef>


Quantiser::Quantiser(size_t qStep): _qStep(qStep){}

void Quantiser::transform(const Block& inBlock, Block& outBlock, Channel channel){
  for(size_t row = 0; row < inBlock.getHeight(channel); ++row){
    for(size_t col = 0; col < inBlock.getWidth(channel); ++col){
      int val = inBlock.getAt(row, col, channel);
      val = (val < 0 ? -1 : 1) * floor(std::abs(val)/_qStep + 0.5);
      outBlock.setAt(row, col, val, channel);
    }
  }
}
