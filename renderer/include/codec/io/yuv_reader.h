/**
* Created by Brent Matthys on 29/03/2025
*/

#pragma once

#include <cstddef>
#include <string>
#include <vector>

std::vector<int> readRawYUVFrame(
  const std::string& path,
  const size_t frame,
  const size_t width,
  const size_t height
);
