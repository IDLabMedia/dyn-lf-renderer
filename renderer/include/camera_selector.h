/**
* Created by Brent Matthys on 25/04/2025
*/

#pragma once

#include "uniforms/camera_uniforms.h"
#include "uniforms/uniforms.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

enum class CameraSelectorStrategy {
  ALL,
  RAY_POINT_DISTANCE
};

class CameraSelector: Uniforms{
private:
  size_t _totalCameras;
  size_t _usedCameras;
  std::vector<int> _selectedCameras;
  std::unique_ptr<CameraUniforms> _cameras;

  static bool _initialized;
  static CameraSelector _instance;

  CameraSelector() = default;
  ~CameraSelector();

  glm::float32* _centerData;
  glm::vec3 getCenterPoint(size_t frame, size_t camera) const;
  double getCameraScore(
    size_t frame, size_t camera, glm::vec3 viewPos, glm::vec3 viewDir
  ) const;

  void selectAll();
  void selectRayPointDistanceCameras(size_t frame, glm::mat4 viewTransform);


public:
  CameraSelector(const CameraSelector&) = delete;
  CameraSelector& operator=(const CameraSelector&) = delete;

  static void initialize(size_t totalCameras, const std::string& inDir, size_t usedCameras);
  static CameraSelector& getInstance();

  /**
  * Select the cameras to use for rendering
  */
  void selectCameras(
    size_t frame,
    glm::mat4 viewTransform,
    CameraSelectorStrategy strategy = CameraSelectorStrategy::RAY_POINT_DISTANCE
  );

  /**
  * Manually set the cameras, the list should at least have
  * getSelectedCount() elements
  */
  void selectCameras(const std::vector<int>& cameras);

  /**
  * Get the amount of selected cameras.
  *
  * Should only be queried after a selectCameras call.
  */
  size_t getSelectedCount() const;

  /**
  * Get the selected camera indices.
  *
  * Should only be queried after a selectCameras call.
  */
  const std::vector<int>& getCameras() const;

  void setUniforms(const ShaderProgram& ShaderProgram) override;
};
