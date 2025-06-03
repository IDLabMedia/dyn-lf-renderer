/**
* Created by Brent Matthys on 30/03/2025
*/

#include <cmath>
#include <cstddef>
#include "codec/encoder/dct.h"
#include "codec/structure/frame.h"

void DCT::transform(const Block& inBlock, Block& outBlock, Channel channel) {
  // F = A * inBlock * B^T
  // shapes:
  // NxM = NxN * NxM * MxM

  size_t N = inBlock.getHeight(channel);
  size_t M = inBlock.getWidth(channel);

  double** A = _matrixProvider.get(N); // A = NxN dct matrix
  double** B = _matrixProvider.get(M); // B = MxM dct matrix

  double tmp[N][M];
  double F[N][M];

  // tmp = A * inBlock
  for(size_t row = 0; row < N; ++row){
    for(size_t col = 0; col < M; ++col){
      tmp[row][col] = 0.0;
      for(size_t k = 0; k < N; ++k){
        tmp[row][col] += A[row][k] * inBlock.getAt(k, col, channel);
      }
    }
  }

  // F = tmp * B^T
  for(size_t row = 0; row < N; ++row){
    for(size_t col = 0; col < M; ++col){
      F[row][col] = 0.0;
      for(size_t k = 0; k < M; ++k){
        F[row][col] += tmp[row][k] * B[col][k]; // B^T[k][col] = B[col][k]
      }
    }
  }

  // write to the output block
  for(size_t row = 0; row < N; ++row){
    for(size_t col = 0; col < M; ++col){
      outBlock.setAt(row, col, static_cast<int>(F[row][col]), channel);
    }
  }
}
