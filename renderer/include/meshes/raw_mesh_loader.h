//
// Created by brent on 12/11/24.
//

#ifndef RAW_MESH_LOADER_H
#define RAW_MESH_LOADER_H

#include "input/buffered_loader.h"
#include "meshes/mesh_loader.h"

class RawMesh: public MeshLoader {
private:
    std::string _inDir;

    BufferedLoader<glm::uint32> _faceBuffer;
    BufferedLoader<glm::float32> _vertexBuffer;

    std::string verticesPath(unsigned int frame) const;
    std::string facesPath(unsigned int frame) const;

public:
    RawMesh(unsigned int width, unsigned int height, std::string inDir, unsigned int totalFrames);

    void fillEBO(unsigned int frame) override;

    void fillVBO(unsigned int frame) override;
};


#endif //RAW_MESH_LOADER_H
