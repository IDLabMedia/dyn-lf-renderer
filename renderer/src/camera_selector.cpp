/**
* Created by Brent Matthys on 25/04/2025
*/
#include "camera_selector.h"
#include "fwd.hpp"
#include "geometric.hpp"
#include "input/reader.h"
#include "matrix.hpp"
#include "shaders/shader_program.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

bool CameraSelector::_initialized = false;
CameraSelector CameraSelector::_instance = CameraSelector();

void CameraSelector::initialize(size_t totalCameras, const std::string& inDir, size_t usedCameras) {
  if (_initialized) {
		std::cerr << "ERROR::CAMERA_SELECTOR::DOUBLE_INIT" << std::endl;
		exit(EXIT_FAILURE);
  }
  _instance._initialized = true;
  _instance._totalCameras = totalCameras;
  _instance._usedCameras = usedCameras;

  glm::uint32 bufferSize = 0;
  _instance._centerData = readRawFileToBuffer<glm::float32>(inDir + "/centers.bin", bufferSize);
  _instance._cameras = std::make_unique<CameraUniforms>(inDir);
}

CameraSelector::~CameraSelector(){
  delete[] _centerData;
}

CameraSelector& CameraSelector::getInstance() {
  if (!_initialized) {
		std::cerr << "ERROR::CAMERA_SELECTOR::NOT_INITIALIZED" << std::endl;
		exit(EXIT_FAILURE);
  }
  return _instance;
}

glm::vec3 CameraSelector::getCenterPoint(size_t frame, size_t camera) const{
  size_t offset = frame * _totalCameras * 3 + camera * 3;
  return glm::vec3(
    _centerData[offset], _centerData[offset + 1], _centerData[offset + 2]
  );
}


double CameraSelector::getCameraScore(size_t frame, size_t camera, glm::vec3 viewPos, glm::vec3 viewDir) const{
  glm::vec3 cam_pos = _cameras->getPositions().at(camera);
  glm::vec3 cam_center_point = getCenterPoint(frame, camera);

  double distToCam = glm::length(cam_pos - viewPos);

  glm::vec3 diff = cam_center_point - viewPos;
  glm::vec3 projected = glm::dot(diff, viewDir) * viewDir;
  float distToRay = glm::length(diff - projected);

  return distToCam / 3; 
}

void CameraSelector::selectAll(){
  _selectedCameras = std::vector<int>(_totalCameras);
  std::iota(_selectedCameras.begin(), _selectedCameras.end(), 0); // {0,1,...,n-1}
}

void CameraSelector::selectRayPointDistanceCameras(size_t frame, glm::mat4 viewTransform){
  viewTransform = glm::inverse(viewTransform);
  glm::vec3 viewPos = glm::vec3(viewTransform[3]);
  glm::vec3 viewDir = -glm::normalize(glm::vec3(viewTransform[2]));

  // get all cameras
  selectAll();

  // order them
  std::sort(
    _selectedCameras.begin(),
    _selectedCameras.end(),
    [this, &frame, &viewPos, &viewDir](const int& lhs, const int& rhs){
      return this->getCameraScore(frame, lhs, viewPos, viewDir) < 
      this->getCameraScore(frame, rhs, viewPos, viewDir);
    }
  );


  // std::cout << "\033[" << _selectedCameras.size() <<  "A";
  // for (auto camera : _selectedCameras) {
  //   std::cout << "\r" << camera << " -> " << getCameraScore(frame, camera, viewTransform) << "                                                                            " << std::endl;
  // }
}


void CameraSelector::selectCameras(const std::vector<int>& cameras){
  if(cameras.size() < getSelectedCount()){
    std::cerr << "ERROR::CAMERA::SELECTOR::SET: got " << cameras.size() << " cameras, but expected at least " << getSelectedCount() << " cameras.";
    exit(EXIT_FAILURE);
  }
  _selectedCameras.assign(cameras.begin(), cameras.begin() + getSelectedCount());
}


void CameraSelector::selectCameras(
  size_t frame, 
  glm::mat4 viewTransform,
  CameraSelectorStrategy strategy
){
  switch (strategy) {
    case CameraSelectorStrategy::ALL: {
      selectAll();
      break;
    }
    case CameraSelectorStrategy::RAY_POINT_DISTANCE: {
      selectRayPointDistanceCameras(frame, viewTransform);
      break;
    }
  }

  // std::cout << "\r" << std::flush;
  // for (auto camera : _selectedCameras) {
  //   std::cout << camera << " " << std::flush;
  // }
}

size_t CameraSelector::getSelectedCount() const{
  return _usedCameras;
}

const std::vector<int>& CameraSelector::getCameras() const{
  return _selectedCameras;
}

void CameraSelector::setUniforms(const ShaderProgram& shaderProgram){
  shaderProgram.setInt("selectedCameraCount", getSelectedCount());
  shaderProgram.setIntArray("selectedCameras", getCameras());
}
