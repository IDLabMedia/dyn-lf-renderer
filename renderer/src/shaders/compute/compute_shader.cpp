//
// Created by brent on 08/03/2025
//

#include <string>
#include "shaders/compute/compute_shader.h"
#include "shaders/shader_program.h"

ComputeShader::ComputeShader(const std::string& shader_file):
  ShaderProgram({ShaderInfo(glCreateShader(GL_COMPUTE_SHADER), shader_file)}){}

void ComputeShader::run(const GLuint dim_x, const GLuint dim_y, const GLuint dim_z) const {
  this->use(); // select this shader program
  glDispatchCompute(dim_x, dim_y, dim_z); // run the comput shader
  glMemoryBarrier( // block until shader completed
    GL_SHADER_STORAGE_BARRIER_BIT |
    GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
    GL_ATOMIC_COUNTER_BARRIER_BIT | 
    GL_BUFFER_UPDATE_BARRIER_BIT
  );
}

