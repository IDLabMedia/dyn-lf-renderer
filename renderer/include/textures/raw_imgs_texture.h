//
// Created by brent on 12/11/24.
//

#ifndef RAW_IMGS_TEXTURE_H
#define RAW_IMGS_TEXTURE_H

#include "textures_loader.h"
#include "input/buffered_loader.h"

class RawImgsTexture: public TexturesLoader {
private:
    std::string _inDir;
    BufferedLoader<glm::uint8> _textureBuffers;
    GLuint cameras;

    std::string computePath(unsigned int frame) const;

public:
    RawImgsTexture(
        std::string inDir,
        const std::vector<glm::vec2>& resolutions,
        unsigned int totalFrames
    );

    void loadFrame(unsigned int frame) override;
};

#endif //RAW_IMGS_TEXTURE_H
