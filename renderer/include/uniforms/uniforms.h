//
// Created by brent on 12/10/24.
//

#ifndef UNIFORMS_H
#define UNIFORMS_H
#include "shaders/shader_program.h"

class Uniforms {
public:
    /**
     * Set the required uniform variables for the shaders.
     *
     * @param shaderProgram The program to set the variables for.
     */
    virtual void setUniforms(const ShaderProgram& shaderProgram) = 0;

    virtual ~Uniforms() = default;
};

#endif //UNIFORMS_H
