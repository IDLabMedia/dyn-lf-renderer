/*
* Created by brent on 22/05/25
*/

#pragma once

#include <glm.hpp>
#include <string>

void writeBufferToPNG(const std::string& filename, glm::uint8* pixels, int width, int height, bool flip=true);
