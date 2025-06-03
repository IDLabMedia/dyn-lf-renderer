//
// Created by brent on 08/03/2025
//

#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include "meshes/mesh_loader.h"
#include "shaders/shader_program.h"
#include <string>

class ComputeShader: public ShaderProgram {
public:
  explicit ComputeShader(const std::string& shader_file);

  /*
  * Run the compute shader.
  */
  void run(GLuint dim_x, GLuint dim_y, GLuint dim_z) const;
};

#endif // !COMPUTE_SHADER_H
