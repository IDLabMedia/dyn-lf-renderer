//
// Created by brent on 22/01/25.
//

#ifndef VOXEL_PCL_LOADER_H
#define VOXEL_PCL_LOADER_H

#include "input/buffered_loader.h"
#include "meshes/color_mesh_loader.h"
#include "meshes/mesh_loader.h"

class VoxelPcl: public ColorMesh{
private:
    BufferedLoader<glm::float32> _voxelBuffer;

    std::string voxelPath(unsigned int frame) const;

public:
    VoxelPcl(unsigned int width, unsigned int height, std::string inDir, unsigned int totalFrames);

    void fillEBO(unsigned int frame) override;

    void fillVBO(unsigned int frame) override;
};


#endif //VOXEL_PCL_LOADER_H
