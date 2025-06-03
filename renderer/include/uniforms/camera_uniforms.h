//
// Created by brent on 12/10/24.
//

#ifndef CAMERA_UNIFORMS_H
#define CAMERA_UNIFORMS_H
#include <nlohmann/json.hpp>

#include "uniforms.h"
#include <glm.hpp>

class CameraUniforms: public Uniforms {
private:
    nlohmann::json _camera_json;

public:
    int getCameraCount() const;
    std::vector<glm::vec2> getPps() const;
    std::vector<glm::vec2> getFocals() const;
    std::vector<glm::vec2> getDepthRange() const;
    std::vector<glm::mat4> getInverseMatrices() const;
    std::vector<glm::vec3> getPositions() const;
    std::vector<glm::vec2> getResolutions() const;

    explicit CameraUniforms(const std::string& inDir);

    void setUniforms(const ShaderProgram& shaderProgram) override;
};

#endif //CAMERA_UNIFORMS_H
