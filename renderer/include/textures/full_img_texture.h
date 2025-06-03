//
// Created by brent on 11/20/24.
//

#ifndef FULL_IMG_TEXTURE_H
#define FULL_IMG_TEXTURE_H

#include "textures_loader.h"
#include <glm.hpp>

class FullImgTexture: public TexturesLoader {
private:
    std::string _inDir;
    unsigned int _cameraCount;
public:
    explicit FullImgTexture(std::string inDir, const std::vector<glm::vec2>& resolutions);

    void loadFrame(unsigned int frame) override;
};

#endif //FULL_IMG_TEXTURE_H
