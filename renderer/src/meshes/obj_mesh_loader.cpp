//
// Created by brent on 11/24/24.
//

#include <glad/glad.h>
#include "meshes/obj_mesh_loader.h"
#include "meshes/mesh_loader.h"

#include <fstream>
#include <iostream>
#include <utility>
#include <sstream>

ObjMesh::ObjMesh(const unsigned int width, const unsigned int height, std::string path): MeshLoader(width, height), _inDir(std::move(path)) {}

std::string ObjMesh::getMeshPath(unsigned int frame) const {
    return _inDir + "/frame_" + std::to_string(frame) + "/mesh.obj";
}


void ObjMesh::fillEBO(unsigned int frame) {
    // open file
    std::ifstream file(getMeshPath(frame));
    if (!file.is_open()) {
        std::cerr << "ERROR::EBO::FAILED_TO_OPEN_OBJ_FILE: " << getMeshPath(frame) << std::endl;
        exit(EXIT_FAILURE);
    }

    // create buffer for faces
    // actually for the indices of the faces
    auto faces = std::vector<unsigned int>();

    // read  all the faces
    std::string line;
    unsigned int idx1, idx2, idx3;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == 'f') {
            std::string indices = line.substr(2);
            std::istringstream ss(indices);
            ss >> idx1 >> idx2 >> idx3;
            faces.push_back(idx1 - 1);
            faces.push_back(idx2 - 1);
            faces.push_back(idx3 - 1);
        }
    }

    file.close();

    // set the number of indices (3 per face)
    _nrIndices = faces.size();

    // bind the data to the EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _nrIndices * sizeof(unsigned int), faces.data(),GL_STATIC_DRAW);

}

void ObjMesh::fillVBO(unsigned int frame) {
    // open file
    std::ifstream file(getMeshPath(frame));
    if (!file.is_open()) {
        std::cerr << "ERROR::EBO::FAILED_TO_OPEN_OBJ_FILE: " << getMeshPath(frame) << std::endl;
        exit(EXIT_FAILURE);
    }

    // create buffer for faces
    auto vertices = std::vector<float>();

    // read  all the faces
    std::string line;
    float x, y, z;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == 'v') {
            std::string indices = line.substr(2);
            std::istringstream ss(indices);
            ss >> x >> y >> z;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    file.close();

    // bind the data to the EBO
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER,  vertices.size() * sizeof(float), vertices.data(),GL_STATIC_DRAW);

    // specify how many floats there are per vertex
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // in the vertex shader, 'texCoords' will be accessed through "layout (location = 0) in vec2 TexCoords;"
    // because we enable vertex attribute 0 (= location 0)
    glEnableVertexAttribArray(0);
}

