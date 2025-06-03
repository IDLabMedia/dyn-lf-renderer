//
// Created by brent on 22/01/25.
//

#include "meshes/voxel_pcl_loader.h"
#include "meshes/color_mesh_loader.h"
#include "shaders/shader_program.h"

VoxelPcl::VoxelPcl(const unsigned int width, const unsigned int height, std::string inDir, const unsigned int totalFrames):
    ColorMesh(width, height, inDir, totalFrames), _voxelBuffer(totalFrames, readRawFileToBuffer<glm::float32>) {
    _voxelBuffer.loadBuffers(0, 30, [this](unsigned int x){return this->voxelPath(x);});
}

std::string VoxelPcl::voxelPath(const unsigned int frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/pcl.vox";
}

void VoxelPcl::fillEBO(const unsigned int frame) {
}

void VoxelPcl::fillVBO(const unsigned int frame) {
    // fetch the vertice data
    glm::float32* vertices = _voxelBuffer.getBuffer(frame, voxelPath(frame));

    _nrPoints = _voxelBuffer.getBufferSize(frame);
    // bind the data to the EBO
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER,  _voxelBuffer.getBufferSize(frame) * sizeof(glm::float32), vertices, GL_STATIC_DRAW);

    // delete the buffer
    _voxelBuffer.clearBuffer(frame);
    // load the next buffer
    _voxelBuffer.loadBuffers(frame + 1, 30, [this](unsigned int x){return this->voxelPath(x);});

    // specify how many floats there are per vertex (second arg)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::float32), (void*)0);
    // in the vertex shader, 'texCoords' will be accessed through "layout (location = 0) in vec2 TexCoords;"
    // because we enable vertex attribute 0 (= location 0)
    glEnableVertexAttribArray(0);

    glEnable(GL_PROGRAM_POINT_SIZE); // enable setting point size in shader
    fillColorVBO(frame);
}
