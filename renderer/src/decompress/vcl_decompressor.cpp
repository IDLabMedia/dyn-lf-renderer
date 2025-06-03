//
// Created by brent on 3/9/25.
//

#include "decompress/vcl_decompressor.h"
#include "profiler.h"
#include <cstddef>

VclDecompressor::VclDecompressor(std::string inDir, const size_t totalFrames, float gridSpacing, size_t bufferedFrames):
    Decompressor(std::move(inDir)),
    _voxelBuffer(totalFrames, readRawFileToBuffer<glm::float32>),
    _hullCubes("compute/hull_cubes.glsl"),
    _bufferedFrames(bufferedFrames),
    _gridSpacing(gridSpacing)
{
    _voxelBuffer.loadBuffers(0, _bufferedFrames,
        [this](const unsigned int x){return this->voxelPath(x);}
        );
}


void VclDecompressor::decompress(
    const size_t frame, std::unique_ptr<MeshLoader>& mesh, std::unique_ptr<TexturesLoader>& texture
) {
    size_t size = 0;
    size = loadVcl(frame);

    PROFILE_SCOPE("Hullcube");
    // perform the convex hull cubes algorithm
    // to compute the faces
    _hullCubes.use(); // use the hullcube shader

    // set inputs
    _vcl.bind(1);
    _hullCubes.setFloat("gridSpacing", _gridSpacing);

    // create outputs
    _faces.bind(2);
    const size_t maxFaces = size * 7;
    if (_faces.size() < maxFaces) {
        _faces.resize(maxFaces);
    }
    _faceCounter.set(0);
    _faceCounter.bind(3);

    // run the compute shader
    _hullCubes.run(size / 3,1,1);
    //std::cout <<  _faceCounter.get() << std::endl;

    // move the vertex data to the VBO and the faces data to the EBO
    mesh->fillVBO(_vcl, size);
    mesh->fillEBO(_faces);
    mesh->accessIndirectDrawCommand().updateCount(_faceCounter);
    glFinish();
}


std::string VclDecompressor::voxelPath(const size_t frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/pcl.vox";
}

size_t VclDecompressor::loadVcl(const size_t frame) {
    PROFILE_SCOPE("Hullcube load");
    // get the frame data
    const glm::float32* vertices = _voxelBuffer.getBuffer(frame, voxelPath(frame));
    // load that data to the GPU on the SSBO
    size_t size;
    size = _voxelBuffer.getBufferSize(frame);


    if (_vcl.size() < size) {
        _vcl.resize(size);
    }
    _vcl.updateData(vertices, size, 0);
    // delete the buffer
    _voxelBuffer.clearBuffer(frame);
    // load the next buffer
    _voxelBuffer.loadBuffers(frame + 1, _bufferedFrames, [this](const unsigned int x){return this->voxelPath(x);});
    glFinish();
    return size;
}
