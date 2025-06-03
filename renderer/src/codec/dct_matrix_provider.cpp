/**
* Created by Brent Matthys on 10/04/2025
*/

#include "codec/dct_matrix_provider.h"

#include <cmath>
#include <mutex>

DCTMatrixProvider::~DCTMatrixProvider(){
  for(auto it : _dctMatrices){ // loop all matrices
    for(size_t row = 0; row < it.first; ++row){
      delete[] it.second[row]; // free rows of matrix
    }
    delete[] it.second; // free matrix
  }
}

double** DCTMatrixProvider::get(size_t size) {
  {
    std::shared_lock<std::shared_mutex> readLock(_mutex); // shared reading
    auto pair = _dctMatrices.find(size);
    if(pair != _dctMatrices.end()) return pair->second; // matrix already present
  }

  { // write the transform matrix
    std::unique_lock<std::shared_mutex> writeLock(_mutex); // exclusive writing
    auto pair = _dctMatrices.find(size);
    if(pair != _dctMatrices.end()) return pair->second; // other thread beat you to computing matrix, so return

    // only this thread has unique lock -> create matrix
    // will be deallocated in destructor
    double** matrix = new double*[size]; // init empty matrix
    for (size_t row = 0; row < size; ++row) {
      matrix[row] = new double[size]; // create empty row
      for (size_t col = 0; col < size; ++col) { // fill row
        double a = sqrt((row == 0 ? 1.0 : 2.0)/size);
        matrix[row][col] = a * cos(((2 * col + 1) * row * M_PI) / (2.0 * size));
      }
    }

    _dctMatrices.emplace(size, matrix); // save matrix to class
  }
  return _dctMatrices.at(size);
}


