//
// Created by brent on 11/20/24.
//

#include "textures/full_img_texture.h"
#include "input/reader.h"

#include <vec2.hpp>

FullImgTexture::FullImgTexture(
  std::string inDir, const std::vector<glm::vec2>& resolutions
): _inDir(std::move(inDir)), _cameraCount(resolutions.size()) {
  _textures.emplace_back(
    "pngTexture",
    false,
    GL_RGB8,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    resolutions.front().x,
    resolutions.front().y,
    resolutions.size()
  );
}

void FullImgTexture::loadFrame(unsigned int frame) {
    // get frame dir
    std::string frameDir = _inDir + "/frame_" + std::to_string(frame);
    glm::uint32 bufferSize;

    // iterate all cameras
    for(GLsizei camera_idx = 0; camera_idx < _cameraCount; camera_idx++) {
        // get frame img
        std::string imgPath = frameDir + "/" + std::to_string(camera_idx) + ".png";
        // load frame img
        glm::uint8* data = readJPEGToBuffer(imgPath, bufferSize);
        // load img to gpu
        _textures.back().updateTextureLayers(data, camera_idx);
        // free img
        free(data);
    }
}
