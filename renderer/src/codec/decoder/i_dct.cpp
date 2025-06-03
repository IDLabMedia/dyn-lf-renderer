/**
* Created by Brent Matthys on 10/04/2025
*/

#include "codec/decoder/i_dct.h"


void IDCT::idct(Block& block, Channel channel) {
  // inverse = A^T * DCT_block * B
  // shapes:
  // NxM = NxN * NxM * MxM

  size_t N = block.getHeight(channel);
  size_t M = block.getWidth(channel);

  double** A = _matrixProvider.get(N); // A = NxN dct matrix
  double** B = _matrixProvider.get(M); // B = MxM dct matrix

  double tmp[N][M];
  double F[N][M];

  // tmp = A^T * inBlock
  for(size_t row = 0; row < N; ++row){
    for(size_t col = 0; col < M; ++col){
      tmp[row][col] = 0.0;
      for(size_t k = 0; k < N; ++k){
        tmp[row][col] += A[k][row] * block.getAt(k, col, channel); // A^T[row][k] = A[k][row]
      }
    }
  }

  // F = tmp * B
  for(size_t row = 0; row < N; ++row){
    for(size_t col = 0; col < M; ++col){
      F[row][col] = 0.0;
      for(size_t k = 0; k < M; ++k){
        F[row][col] += tmp[row][k] * B[k][col];
      }
    }
  }

  // write to the output block
  for(size_t row = 0; row < N; ++row){
    for(size_t col = 0; col < M; ++col){
      block.setAt(row, col, static_cast<int>(F[row][col]), channel);
    }
  }
}

