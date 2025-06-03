//
// Created by brent on 12/10/24.
//

#include <glad/glad.h>
#include "meshes/dummy_mesh_loader.h"
#include "meshes/mesh_loader.h"

DummyMesh::DummyMesh(const unsigned int width, const unsigned int height): MeshLoader(width, height) {}

void DummyMesh::fillEBO(unsigned int frame) {
    unsigned int faces[] = {
      0, 1, 2
    };

    // set the number of indices (3 per face)
    _nrIndices = 3;

    // bind the data to the EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _nrIndices * sizeof(unsigned int), faces, GL_STATIC_DRAW);
}


void DummyMesh::fillVBO(unsigned int frame) {
    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    // bind the data to the EBO
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER,  sizeof(vertices), vertices, GL_STATIC_DRAW);

    // specify how many floats there are per vertex (second arg)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // in the vertex shader, 'texCoords' will be accessed through "layout (location = 0) in vec2 TexCoords;"
    // because we enable vertex attribute 0 (= location 0)
    glEnableVertexAttribArray(0);
}
