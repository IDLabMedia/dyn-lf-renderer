//
// Created by brent on 11/24/24.
//

#ifndef OBJ_MESH_LOADER_H
#define OBJ_MESH_LOADER_H

#include "meshes/mesh_loader.h"

class ObjMesh: public MeshLoader {
private:
    std::string _inDir;

    std::string getMeshPath(unsigned int frame) const;

public:
    ObjMesh(unsigned int width, unsigned int height, std::string path);

    void fillEBO(unsigned int frame) override;

    void fillVBO(unsigned int frame) override;

};

#endif //OBJ_MESH_LOADER_H
