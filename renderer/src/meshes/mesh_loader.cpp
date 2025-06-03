//
// Created by brent on 11/22/24.
//

#include <glad/glad.h>
#include "meshes/mesh_loader.h"

#include <memory>

MeshLoader::MeshLoader(const unsigned int width, const unsigned int height, bool headless): _width(width), _height(height) {
    if(headless){
      glGenFramebuffers(1, &_FBO);
      glBindFramebuffer(GL_FRAMEBUFFER, _FBO);
    }else{
      // by binding framebuffer 0, if we run a shader, it writes to the screen texture (actually GlfW window)
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    // define the resolution of the screen texture
    glViewport(0, 0, width, height);
    // define the clear color (black)
    glClearColor(0, 0, 0, 1);

	glGenVertexArrays(1, &_VAO);
	glGenBuffers(1, &_VBO);
	glGenBuffers(1, &_EBO);
	glBindVertexArray(_VAO);
}

void MeshLoader::bindVAOAndDraw() const{
  // bind the VAO, and thus the VBO and EBO
	glBindVertexArray(_VAO);
	// draw the vertices, by using the indices in the EBO
  if(_nrIndices == -1){ // draw indirect
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, _indirectCommand.getBufferId());
    glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0);
  }if(_nrPoints != 0){ // render point cloud
      glDrawArrays(GL_POINTS, 0, _nrPoints); // Number of vertices
      glPointSize(4.55);
  }else{ // render mesh
	    glDrawElements(GL_TRIANGLES, _nrIndices, GL_UNSIGNED_INT, 0); // Number of faces * 3
  }
}

void MeshLoader::renderMesh(const std::unique_ptr<TexturesLoader>& textureLoader) const {
	// render to screen
  glViewport(0,0, _width, _height);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	// clear the screen buffer and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// bind input texture
	if (textureLoader) {
		textureLoader->bindTextures();
	}
  bindVAOAndDraw();
}

void MeshLoader::fillEBO(unsigned int frame) {
    // Leave empty, should already be set manually
}
void MeshLoader::fillEBO(const void *data, const GLuint size) {
    _nrIndices = size;
    // bind the data to the EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size * sizeof(glm::uint32), data, GL_STATIC_DRAW);
}


void MeshLoader::fillVBO(unsigned int frame) {
    // Leave empty, should already be set manually
}
void MeshLoader::fillVBO(const void *data, const GLuint size) {
	//_nrPoints = size;
    // bind the data to the VBO
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER,  size * sizeof(glm::float32), data, GL_STATIC_DRAW);

    // specify how many floats there are per vertex (second arg)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(glm::float32), (void*)0);
    // in the vertex shader, 'texCoords' will be accessed through "layout (location = 0) in vec2 TexCoords;"
    // because we enable vertex attribute 0 (= location 0)
    glEnableVertexAttribArray(0);
}


IndirectDrawCommand& MeshLoader::accessIndirectDrawCommand(){
  return _indirectCommand;
}

MeshLoader::~MeshLoader() {
	glDeleteVertexArrays(1, &_VAO);
	glDeleteBuffers(1, &_VBO);
	glDeleteBuffers(1, &_EBO);
}
