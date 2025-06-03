//
// Created by brent on 12/10/24.
//

#ifndef DUMMY_MESH_LOADER_H
#define DUMMY_MESH_LOADER_H

#include "meshes/mesh_loader.h"

class DummyMesh: public MeshLoader {
public:
    DummyMesh(unsigned int width, unsigned int height);

    void fillEBO(unsigned int frame) override;

    void fillVBO(unsigned int frame) override;
};
#endif //DUMMY_MESH_LOADER_H
