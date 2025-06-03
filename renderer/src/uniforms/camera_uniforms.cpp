//
// Created by brent on 12/10/24.
//

#include "uniforms/camera_uniforms.h"

#include <iostream>

#include "input/json_reader.h"


CameraUniforms::CameraUniforms(const std::string &inDir) {
    _camera_json = readJsonFile(inDir + "/cameras.json");
    if (!_camera_json.is_array()) {
        std::cerr << "ERROR::JSON::CAMERAS::NO_ARRAY " << std::endl;
        exit(EXIT_FAILURE);
    }
}

int CameraUniforms::getCameraCount() const {
    return _camera_json.size();
}

std::vector<glm::vec2> CameraUniforms::getResolutions() const {
    std::vector<glm::vec2> resolutions;
    for (const auto& camera: _camera_json) {
        std::string key = "resolution";
        resolutions.emplace_back((int)camera[key][0], (int)camera[key][1]);
    }
    return resolutions;
}

std::vector<glm::vec2> CameraUniforms::getPps() const {
     std::vector<glm::vec2> pps;
    for (const auto& camera: _camera_json) {
        std::string key = "pp";
        pps.emplace_back((float)camera[key][0], (float)camera[key][1]);
    }
    return pps;
}

std::vector<glm::vec2> CameraUniforms::getFocals() const {
    std::vector<glm::vec2> focals;
    for (const auto& camera: _camera_json) {
        std::string key = "focal";
        focals.emplace_back((float)camera[key][0], (float)camera[key][1]);
    }
    return focals;
}

std::vector<glm::vec2> CameraUniforms::getDepthRange() const {
    std::vector<glm::vec2> depth_range;
    for (const auto& camera: _camera_json) {
        std::string key = "depth_range";
        depth_range.emplace_back((float)camera[key][0], (float)camera[key][1]);
    }
    return depth_range;
}

std::vector<glm::mat4> CameraUniforms::getInverseMatrices() const {
    std::vector<glm::mat4> invModels;
    for (const auto& camera: _camera_json) {
        std::string key = "inv_model";
        invModels.emplace_back(
          (float)camera[key][0][0],(float)camera[key][1][0], (float)camera[key][2][0], (float)camera[key][3][0],
          (float)camera[key][0][1],(float)camera[key][1][1], (float)camera[key][2][1], (float)camera[key][3][1],
          (float)camera[key][0][2],(float)camera[key][1][2], (float)camera[key][2][2], (float)camera[key][3][2],
          (float)camera[key][0][3],(float)camera[key][1][3], (float)camera[key][2][3], (float)camera[key][3][3]
        );
    }
    return invModels;
}

std::vector<glm::vec3> CameraUniforms::getPositions() const {
    std::vector<glm::vec3> positions;
    for (const auto& camera: _camera_json) {
        std::string key = "position";
        positions.emplace_back((float)camera[key][0], (float)camera[key][1], (float)camera[key][2]);
    }
    return positions;
}


void CameraUniforms::setUniforms(const ShaderProgram &shaderProgram) {
    shaderProgram.setInt("cameras", this->getCameraCount());
    shaderProgram.setVec2Array("resolutions", this->getResolutions());
    shaderProgram.setVec2Array("pps", this->getPps());
    shaderProgram.setVec2Array("focals", this->getFocals());
    shaderProgram.setVec2Array("depth_range", this->getDepthRange());
    shaderProgram.setMat4Array("invMatrices", this->getInverseMatrices());
    shaderProgram.setVec3Array("positions", this->getPositions());
}
