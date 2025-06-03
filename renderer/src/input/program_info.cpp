//
// Created by brent on 12/10/24.
//

#include <glad/glad.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>

#include "input/program_info.h"

#include "camera_selector.h"
#include "decompress/vcl_decompressor.h"

#include "meshes/color_mesh_loader.h"
#include "meshes/dummy_mesh_loader.h"
#include "meshes/raw_mesh_loader.h"
#include "meshes/voxel_pcl_loader.h"

#include "shaders/depth_shader.h"
#include "textures/jpeg_texture.h"
#include "textures/raw_imgs_texture.h"
#include "textures/codec_texture.h"
#include "textures/yuv_texture.h"

#include "uniforms/camera_uniforms.h"

ProgramInfo::ProgramInfo(int argc, char *argv[]) {
    // check if at least 1 argument is passed
    if (argc < 2) {
        writeHelpMsg(std::cerr);
        exit(EXIT_FAILURE);
    }

    // check if help message was passed
    for (unsigned int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            writeHelpMsg(std::cout);
            exit(EXIT_SUCCESS);
        }
    }

    // get input dir
    _inDir = argv[1];

    // handle optional arguments
    for (unsigned int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            _meshType = this->meshTypeFromString(argv[++i]);
        } else if (arg == "-f" && i + 1 < argc) {
            _fragmentType = this->fragmentTypeFromString(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            _windowWidth = std::stoul(argv[++i]);
        } else if (arg == "-h" && i + 1 < argc) {
            _windowHeight = std::stoul(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            _totalFrames = std::stoul(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            _gridSpacing = std::stof(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            _usedCameras = std::stof(argv[++i]);
        } else if (arg == "--headless" && i + 2 < argc){
            _headless = true;
            _headlessViewpoint = std::stoul(argv[++i]);
            _headlessFrame = std::stoul(argv[++i]);
        } else {
            std::cerr << "Unknown option " << arg << " or invalid arguments passed for option." << std::endl;
            writeHelpMsg(std::cerr);
            exit(EXIT_FAILURE);
        }
    }

}

MeshType ProgramInfo::meshTypeFromString(const std::string &str) const {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    // find the correct value
    if (lowerStr == "dummy") {
        return MeshType::DUMMY;
    }
    if (lowerStr == "raw") {
        return MeshType::RAW;
    }
    if (lowerStr == "voxel") {
        return MeshType::VOXEL;
    }
    if (lowerStr == "vmesh" || lowerStr == "voxel-mesh") {
        return MeshType::VMESH;
    }

    // no match found
    std::cerr << "Unknown mesh type: " << lowerStr << std::endl;
    writeHelpMsg(std::cerr);
    exit(EXIT_FAILURE);
}

FragmentType ProgramInfo::fragmentTypeFromString(const std::string &str) const {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    // find the correct value
    if (lowerStr == "dummy") {
        return FragmentType::DUMMY;
    }
    if(lowerStr == "codec"){
        return FragmentType::CODEC;
    }
    if (lowerStr == "raw") {
        return FragmentType::RAW;
    }
    if (lowerStr == "jpeg") {
        return FragmentType::JPEG;
    }
    if (lowerStr == "yuv") {
        return FragmentType::YUV;
    }
    if (lowerStr == "ambient") {
        return FragmentType::AMBIENT;
    }

    // no match found
    std::cerr << "Unknown fragment type: " << lowerStr << std::endl;
    writeHelpMsg(std::cerr);
    exit(EXIT_FAILURE);
}


void ProgramInfo::writeHelpMsg(std::ostream &out) const {
    out << "Usage: RTDLF [--help] input-dir [-m mesh-type] [-f fragment-type] [-w width] [-h height]" << std::endl;
    out << std::endl;

    out << "Required arguments:" << std::endl;
    out << "    input-dir: The input directory";
    out << std::endl;

    out << "Optional arguments:" << std::endl;
    out << "    --help: Show this help and exit" << std::endl;
    out << "    -m mesh-type: Specifies the mesh type (default: dummy)" << std::endl;
    out << "    -f fragment-type: Specifies the fragment type (default: dummy)" << std::endl;
    out << "    -w width: Specifies the width (default: 960)" << std::endl;
    out << "    -h height: Specifies the height (default: 540)" << std::endl;
    out << std::endl;
}

std::vector<ShaderInfo> ProgramInfo::getShaders() const {
    std::vector<ShaderInfo> shaders;

    // always first add vertex shaders
    switch (_meshType) {
        case MeshType::DUMMY: {
            shaders.emplace_back(glCreateShader(GL_VERTEX_SHADER),  "dummy.vs");
            break;
        }
        case MeshType::RAW: {
            shaders.emplace_back(glCreateShader(GL_VERTEX_SHADER), "raw.vs");
            break;
        }
        case MeshType::VOXEL: {
            shaders.emplace_back(glCreateShader(GL_VERTEX_SHADER), "color.vs");
            break;
        }
        case MeshType::VMESH: {
            shaders.emplace_back(glCreateShader(GL_VERTEX_SHADER), "color.vs");
            break;
        }
    }
    switch (_fragmentType) {
        case FragmentType::DUMMY: {
            shaders.emplace_back(glCreateShader(GL_FRAGMENT_SHADER), "dummy.fs");
            break;
        }
        case FragmentType::YUV:
        case FragmentType::CODEC: {
            shaders.emplace_back(glCreateShader(GL_FRAGMENT_SHADER), "yuv.fs");
            break;
        }
        case FragmentType::JPEG: // jpegs are decompressed on cpu, so it can use the raw fragment shader
        case FragmentType::RAW: {
            shaders.emplace_back(glCreateShader(GL_FRAGMENT_SHADER), "raw.fs");
            break;
        }
        case FragmentType::AMBIENT:{
            shaders.emplace_back(glCreateShader(GL_FRAGMENT_SHADER), "ambient.fs");
            break;
        }
    }
    return shaders;
}

std::vector<std::unique_ptr<Decompressor>> ProgramInfo::getDecompressors() const {
    std::vector<std::unique_ptr<Decompressor>> decompressors;
    switch (_meshType) {
        case MeshType::VMESH: {
            decompressors.push_back(std::make_unique<VclDecompressor>(_inDir, _totalFrames, _gridSpacing, _headless ? 0 : 5));
            break;
        }
        default: /*No decompressor*/;
    }
    switch (_fragmentType) {
        default: /*No decompressor*/;
    }
    return decompressors;
}

std::unique_ptr<DepthShader> ProgramInfo::getDepthShader() const{
  return std::make_unique<DepthShader>(_inDir, "depthTexture");
}

std::unique_ptr<MeshLoader> ProgramInfo::getMeshLoader() const {
    switch (_meshType) {
        case MeshType::RAW: return std::make_unique<RawMesh>(_windowWidth, _windowHeight, _inDir, _totalFrames);
        case MeshType::VOXEL: return std::make_unique<VoxelPcl>(_windowWidth, _windowHeight, _inDir, _totalFrames);
        case MeshType::VMESH: return std::make_unique<ColorMesh>(
          _windowWidth, _windowHeight, _inDir, _totalFrames, _headless ? 0 : 5
        );
        default: {
            return std::make_unique<DummyMesh>(_windowWidth, _windowHeight);
        }
    }
}


std::unique_ptr<TexturesLoader> ProgramInfo::getTexturesLoader() const {
    const auto& resolutions = CameraUniforms(_inDir).getResolutions();
    std::unique_ptr<TexturesLoader> textureLoader = nullptr;

    switch (_meshType) {
        default: /*No textures*/;
    }
    switch (_fragmentType) {
        case FragmentType::RAW: {
            textureLoader = std::make_unique<RawImgsTexture>(
                _inDir, resolutions, _totalFrames);
            break;
        }
        case FragmentType::CODEC: {
            textureLoader = std::make_unique<CodecTexture>(
                _inDir, resolutions.front().x, resolutions.front().y
            );

          break;
        }
        case FragmentType::YUV: {
            textureLoader = std::make_unique<YUVTexture>(
                _inDir, resolutions.front().x, resolutions.front().y, _totalFrames, _headless
            );
            break;
        }
        case FragmentType::JPEG: {
            textureLoader = std::make_unique<JPEGTexture>(
                _inDir, resolutions, _totalFrames);
            break;
        }
        default: /*No textures*/;
    }
    return textureLoader;
}

std::vector<std::unique_ptr<Uniforms>> ProgramInfo::getUniformsList() const {
    std::vector<std::unique_ptr<Uniforms>> uniformsList;
    switch (_meshType) {
        default: /*No uniforms to set*/;
    }
    switch (_fragmentType) {
        case FragmentType::JPEG:
        case FragmentType::CODEC:
        case FragmentType::YUV:
        case FragmentType::RAW:
        default: {
            uniformsList.emplace_back(std::make_unique<CameraUniforms>(_inDir));
            break;
        };
    }
    // always pass the used cameras to the gpu
    auto cameraInfo = std::make_unique<CameraUniforms>(_inDir); 
    CameraSelector::initialize(cameraInfo->getCameraCount(), _inDir, _usedCameras);
    return uniformsList;
}

glm::mat4 ProgramInfo::getBasicView() const {
  auto matriches = CameraUniforms(_inDir).getInverseMatrices();
  int cam = _headlessViewpoint == -1 ? 0 : _headlessViewpoint;
  return matriches.at(cam);
}

