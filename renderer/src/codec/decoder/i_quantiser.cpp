/**
* Created by Brent Matthys on 10/04/2025
*/

#include "codec/decoder/i_quantiser.h"
#include <cmath>

IQuantiser::IQuantiser(size_t qStep): _qStep(qStep){}

void IQuantiser::iQuantize(Block& block, Channel channel){
  for(size_t row = 0; row < block.getHeight(channel); ++row){
    for(size_t col = 0; col < block.getWidth(channel); ++col){
      block.setAt(row, col, block.getAt(row, col, channel) * _qStep, channel);
    }
  }
}

