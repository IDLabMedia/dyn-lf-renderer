/**
* Created by Brent Matthys on 10/04/2025
*/

#pragma once

#include "codec/dct_matrix_provider.h"
#include "codec/structure/frame.h"
class IDCT {
private:
  DCTMatrixProvider _matrixProvider = DCTMatrixProvider();
public:
  void idct(Block& block, Channel channel);
};


