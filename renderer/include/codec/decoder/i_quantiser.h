/**
* Created by Brent Matthys on 10/04/2025
*/

#pragma once

#include "codec/structure/frame.h"
#include <cstddef>

class IQuantiser{
private:
  size_t _qStep;
public:
  explicit IQuantiser(size_t qStep);
  void iQuantize(Block& block, Channel channel);
};
