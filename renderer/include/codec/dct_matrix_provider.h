/**
* Created by Brent Matthys on 10/04/2025
*/

#pragma once

#include <shared_mutex>
#include <unordered_map>

class DCTMatrixProvider {
private:
  std::unordered_map<size_t, double**> _dctMatrices;
  std::shared_mutex _mutex;

public:
  double** get(size_t size);
  ~DCTMatrixProvider();
};


