//
// Created by brent on 12/11/24.
//

#include "meshes/raw_mesh_loader.h"
#include "fwd.hpp"
#include "input/reader.h"

RawMesh::RawMesh(const unsigned int width, const unsigned int height, std::string inDir, const unsigned int totalFrames):
    MeshLoader(width, height),
    _inDir(std::move(inDir)),
    _faceBuffer(totalFrames, readRawFileToBuffer<glm::uint32>),
    _vertexBuffer(totalFrames, readRawFileToBuffer<glm::float32>) {}

std::string RawMesh::verticesPath(const unsigned int frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/vertices.raw";
}

std::string RawMesh::facesPath(const unsigned int frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/faces.raw";
}

void RawMesh::fillEBO(const unsigned int frame) {
    // fetch the face indices and set the nrIndices
    glm::uint32* faces = _faceBuffer.getBuffer(frame, facesPath(frame));
    _nrIndices = _faceBuffer.getBufferSize(frame);


    // bind the data to the EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _nrIndices * sizeof(glm::uint32), faces, GL_STATIC_DRAW);

    // free the buffer
    _faceBuffer.clearBuffer(frame);
    // load the next buffer
    _faceBuffer.loadBuffer(frame + 1, facesPath(frame + 1));
}

void RawMesh::fillVBO(const unsigned int frame) {
    // fetch the vertice data
    glm::float32* vertices = _vertexBuffer.getBuffer(frame, verticesPath(frame));

    // bind the data to the VBO
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER,  _vertexBuffer.getBufferSize(frame) * sizeof(glm::float32), vertices, GL_STATIC_DRAW);

    // delete the buffer
    _vertexBuffer.clearBuffer(frame);
    // load the next buffer
    _vertexBuffer.loadBuffer(frame + 1, verticesPath(frame + 1));

    // specify how many floats there are per vertex (second arg)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::float32), (void*)0);
    // in the vertex shader, 'texCoords' will be accessed through "layout (location = 0) in vec2 TexCoords;"
    // because we enable vertex attribute 0 (= location 0)
    glEnableVertexAttribArray(0);
}
