//
// Created by brent on 11/22/24.
//

#ifndef MESH_LOADER_H
#define MESH_LOADER_H


#include "gpu_wrappers/indirect_command.h"
#include "textures/textures_loader.h"
#include "gpu_wrappers/ssbo.h"

#include <memory>


class MeshLoader {
protected:
    GLuint _VAO, _VBO, _EBO, _FBO = 0;
    unsigned int _nrIndices = 0; // total indices (=3*faces) (used when rendering mesh)
    unsigned int _nrPoints = 0; // total points (used when rendering point cloud)

    unsigned int _width;
    unsigned int _height;

    IndirectDrawCommand _indirectCommand;

public:
    /**
     * Fill the EBO buffer, so that triangles are defined.
     *
     * !!This should update the _nrIndices field to reflect the amount
     * of indices currently in the EBO!!
     *
     * EBO (Element Buffer Objects) is a buffer that keeps
     * triplets of indices of vertices (in the VBO) to define
     * triangles to be rendered.
     *
     * This function can be left empty when rendering point clouds.
     *
     * @param frame The frame to load the mesh for
     */
    virtual void fillEBO(unsigned int frame);
    void fillEBO(const void *data, GLuint size);

    template <typename T>
    void fillEBO(const SSBO<T>& ssbo, const size_t size) {
        // fillEBO(nullptr, size);
        // glInvalidateBufferData(_EBO);
        // glCopyNamedBufferSubData(ssbo.id(), _EBO, 0,0, size * sizeof(T));


        // directly use ssbo as ebo
        _nrIndices = size;
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ssbo.id());
    }

    template <typename T>
    void fillEBO(const SSBO<T>& ssbo) {
        // directly use ssbo as ebo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ssbo.id());
        _nrIndices = -1; // indicate that draw should be indirect
    }


    /**
     * Fill the VBO buffer, so that the shader can render vertices.
     *
     * !!This should update the _nrPoints field to reflect the amount
     * of points currently in the VBO, when rendering point clouds!!
     *
     * VBO (Vertex Buffer Objects) is a buffer that keeps all
     * data that defines a single vertex.
     *
     * @param frame The frame to load the mesh for
     */
    virtual void fillVBO(unsigned int frame);
    void fillVBO(const void* data, GLuint size);


    template <typename T>
    void fillVBO(const SSBO<T>& ssbo, const GLuint size) {
        fillVBO(nullptr, size);
        glInvalidateBufferData(_VBO);
        // load ssbo to vbo
        glCopyNamedBufferSubData(ssbo.id(), _VBO, 0,0, size * sizeof(T));

        // glBindBuffer(GL_ARRAY_BUFFER, ssbo.id());
        // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::float32), (void*)0);
        // glEnableVertexAttribArray(0);
    }

    IndirectDrawCommand& accessIndirectDrawCommand();

    /*
     * Create the mesh loader.
     *
     * @param width The width of the viewport
     * @param height The height of the viewport
     */
    MeshLoader(unsigned int width, unsigned int height, bool headless = false);

    void bindVAOAndDraw() const;

    /**
     * Render the mesh
     */
    void renderMesh(const std::unique_ptr<TexturesLoader>& texture) const;
    

    /**
     * MeshLoader will free the VAO, VBO and EBO
     * child classes should delete their own buffers
     * they used to fill those shader buffers.
     */
    virtual ~MeshLoader();
};

#endif //MESH_LOADER_H
