/**
* Created by Brent Matthys on 12/04/2025
*/

#include "textures/codec_texture.h"
#include "camera_selector.h"
#include "codec/structure/frame.h"
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>


CodecTexture::CodecTexture(std::string inDir, GLuint frameWidth, GLuint frameHeight): _decoder(Decoder(inDir)){
  int usedCameras = CameraSelector::getInstance().getSelectedCount();
  _textures.emplace_back(
    "yTexture",
    false,
    GL_R8,
    GL_RED,
    GL_UNSIGNED_BYTE,
    frameWidth,
    frameHeight,
    usedCameras
  );
  _textures.emplace_back(
    "uTexture",
    false,
    GL_R8,
    GL_RED,
    GL_UNSIGNED_BYTE,
    frameWidth / 2,
    frameHeight / 2,
    usedCameras
  );
  _textures.emplace_back(
    "vTexture",
    false,
    GL_R8,
    GL_RED,
    GL_UNSIGNED_BYTE,
    frameWidth / 2,
    frameHeight / 2,
    usedCameras
  );
}

void CodecTexture::loadFrame(unsigned int frame){
  auto cams = CameraSelector::getInstance().getCameras();
  std::vector<size_t> cameras(cams.begin(), cams.end());
  std::vector<Frame> frames = _decoder.decode(frame, cameras);

  for(GLuint i = 0; i < 2; ++i){
    _textures[0].updateTextureLayers(frames[i].getY().data(), i);
    _textures[1].updateTextureLayers(frames[i].getU().data(), i);
    _textures[2].updateTextureLayers(frames[i].getV().data(), i);
  }
}

