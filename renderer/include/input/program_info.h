//
// Created by brent on 12/10/24.
//

#ifndef PROGRAM_INFO_H
#define PROGRAM_INFO_H
#include <cstddef>
#include <string>
#include <vector>

#include "decompress/decompressor.h"
#include "meshes/mesh_loader.h"
#include "shaders/depth_shader.h"
#include "shaders/shader_program.h"
#include "uniforms/uniforms.h"

enum class MeshType {
    DUMMY,
    RAW,
    VOXEL,
    VMESH
};

enum class FragmentType {
    DUMMY,
    CODEC,
    RAW,
    JPEG,
    YUV,
    AMBIENT,
};

class ProgramInfo {
private:
    std::string _inDir;

    MeshType _meshType = MeshType::VMESH;
    FragmentType _fragmentType = FragmentType::YUV;

    unsigned int _windowWidth = 1280;
    unsigned int _windowHeight = 720;

    nlohmann::json _metadataJson;

    size_t _usedCameras = 3;

    bool _headless = false;
    size_t _headlessFrame = 0;
    std::string _headlessOutPath;

    float getGridSpacing() const;
    void writeHelpMsg(std::ostream& out) const;
    MeshType meshTypeFromString(const std::string& str) const;
    FragmentType fragmentTypeFromString(const std::string& str) const;
public:
    ProgramInfo(int argc, char* argv[]);

    unsigned int getWindowWidth() const {return _windowWidth;}
    unsigned int getWindowHeight() const {return _windowHeight;}
    size_t getTotalFrames() const;
    bool isHeadless() const {return _headless;}
    size_t headlessFrame() const {return _headlessFrame;}
    const std::string& getHeadlessOutPath() const {return _headlessOutPath;}

    std::vector<ShaderInfo> getShaders() const;
    std::vector<std::unique_ptr<Decompressor>> getDecompressors() const;
    std::unique_ptr<DepthShader> getDepthShader() const;
    std::unique_ptr<MeshLoader> getMeshLoader() const;
    std::unique_ptr<TexturesLoader> getTexturesLoader() const;
    std::vector<std::unique_ptr<Uniforms>> getUniformsList() const;
    glm::mat4 getBasicView() const;
};

#endif //PROGRAM_INFO_H
