/*
* Created by brent on 22/05/25
*/

#include "output/writer.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION // SHOULD ONLY BE DEFINED ONCE
#include "stb_image_write.h"


void writeBufferToPNG(const std::string& filename, glm::uint8* pixels, int width, int height, bool flip){
  if(flip){
    int rowSize = width * 4;
    uint8_t* tempRow = new uint8_t[rowSize];

    for (int y = 0; y < height / 2; ++y) {
      uint8_t* rowTop = pixels + y * rowSize;
      uint8_t* rowBottom = pixels + (height - 1 - y) * rowSize;

      memcpy(tempRow, rowTop, rowSize);
      memcpy(rowTop, rowBottom, rowSize);
      memcpy(rowBottom, tempRow, rowSize);
    }

    delete[] tempRow;
  }
  stbi_write_png(filename.c_str(), width, height, 4, pixels, width * 4);
}


void flipImageVertically(std::vector<unsigned char>& pixels, int width, int height) {
  int rowSize = width * 4; // 4 bytes per pixel (RGBA)
  std::vector<unsigned char> tempRow(rowSize);

  for (int y = 0; y < height / 2; ++y) {
    unsigned char* rowTop = &pixels[y * rowSize];
    unsigned char* rowBottom = &pixels[(height - 1 - y) * rowSize];

    // Swap rows
    memcpy(tempRow.data(), rowTop, rowSize);
    memcpy(rowTop, rowBottom, rowSize); 
    memcpy(rowBottom, tempRow.data(), rowSize); 
  } 
}
