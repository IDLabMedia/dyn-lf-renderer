//
// Created by brent on 11/20/24.
//


#include "application.h"
#include "GLFW/glfw3.h"
#include "camera_selector.h"
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <ext/matrix_transform.hpp>
#include "ext/matrix_float4x4.hpp"
#include "gpu_wrappers/atomic_counter.h"
#include "input/program_info.h"
#include "output/writer.h"
#include "profiler.h"
//#include <gtx/euler_angles.hpp>



void Application::initWindow() {
    // check if window already exists
    if (_window) {
        std::cerr << "ERROR::APPLICATION::CANNOT_INIT_WINDOW_TWICE" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize GLFW
    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "ERROR::GLFW::FAILED_TO_INITIALIZE" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set GLFW to not create an OpenGL context (version 460 core) (also at the top of each shader file)
    if(_headless){
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Create a GLFW window
    _window = glfwCreateWindow(_windowWidth, _windowHeight, "RTDLF", nullptr, nullptr);
    if (!_window) {
        std::cerr << "ERROR::GLFW::FAILED_TO_CREATE_WINDOW" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(_window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "ERROR::GLAD::FAILED_TO_INITIALIZE" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // disable Vsync
    glfwSwapInterval(0);

    // Enable depth testing in OpenGL
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // By default, OpenGL assumes the dimensions of each texture are divisible by 4
    // otherwise, we need to change the alignment
    if ((int)_windowWidth % 4 != 0 || (int)_windowHeight% 4 != 0) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
}

void Application::initShaderProgram(const std::vector<ShaderInfo> &shaders) {
    // check if the window is created
    if (!_window) {
        std::cerr << "ERROR::APPLICATION::SHADERPROGRAM::GLFWWINDOW_DOESNT_EXIST" << std::endl;
        std::cerr << "Ensure a GLFWwindow is created before creating the shader program" << std::endl;
        exit(EXIT_FAILURE);
    }
    _shaderProgram = std::make_unique<ShaderProgram>(shaders);
}

void Application::setMesh(std::unique_ptr<MeshLoader> meshLoader) {
    // check if shader is created
    if (!_shaderProgram) {
        std::cerr << "ERROR::APPLICATION::MESH::SHADER_DOESNT_EXIST" << std::endl;
        std::cerr << "Ensure a the shader program is created before creating mesh loader" << std::endl;
        exit(EXIT_FAILURE);
    }
    _mesh = std::move(meshLoader);
}

void Application::addTexture(std::unique_ptr<TexturesLoader> textureLoader) {
    // check if a mesh is set
    if (!_mesh) {
        std::cerr << "ERROR::APPLICATION::TEXTURE::MESH_DOESNT_EXIST" << std::endl;
        std::cerr << "Ensure a mesh is set before adding textures" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (!textureLoader) { // ensure a texture is used
        return;
    }
    _textureLoader = std::move(textureLoader);
    _textureLoader->assignTexturesToShader(*_shaderProgram);
}

void Application::setUniforms(const std::unique_ptr<Uniforms>& uniforms) {
    // check if shader is created
    if (!_shaderProgram) {
        std::cerr << "ERROR::APPLICATION::UNIFORMS::SHADER_DOESNT_EXIST" << std::endl;
        std::cerr << "Ensure a the shader program is created before setting the uniforms" << std::endl;
        exit(EXIT_FAILURE);
    }
    uniforms->setUniforms(*_shaderProgram);

}

glm::mat4 Application::getViewMatrix() {
  float yawRad = glm::radians(_yaw);
  float pitchRad = glm::radians(_pitch);

  glm::mat4 rotation = glm::mat4(1.0f);
  rotation = glm::rotate(rotation, glm::radians(_yaw), glm::vec3(0.0f, 1.0f, 0.0f));   // yaw
  rotation = glm::rotate(rotation, glm::radians(_pitch), glm::vec3(1.0f, 0.0f, 0.0f)); // pitch

  return rotation * glm::translate(_basicView, _accumMovement);
}

void Application::setViewMatrix(){
  _shaderProgram->setMat4("view", getViewMatrix());
}


void Application::initialize(const ProgramInfo &programInfo) {
    _headless = programInfo.isHeadless();
    _headlessViewpoint = programInfo.headlessViewpoint();
    _windowWidth = programInfo.getWindowWidth();
    _windowHeight = programInfo.getWindowHeight();
    this->initWindow();
    _totalFrames = programInfo.getTotalFrames();
    
    // create the shader program // AND USE IT!! otherwise textures and uniforms won't be set
    this->initShaderProgram(programInfo.getShaders());
    _shaderProgram->use();

    // set the basicView
    _basicView = programInfo.getBasicView();
    setViewMatrix();


    // load the shader uniforms (=shader constants)
    for (const auto& uniforms: programInfo.getUniformsList()) {
        this->setUniforms(uniforms);
    }
    CameraSelector::getInstance().selectCameras(0, getViewMatrix());

    if(_headlessViewpoint != -1){
        // prevent the use of the viewpoint camera when generating viewpoint
        auto cameras = CameraSelector::getInstance().getCameras();
        std::vector<int> filtered;
        std::copy_if(
            cameras.begin(), cameras.end(), std::back_inserter(filtered),
            [this](int x){return x != _headlessViewpoint;}
        );
        CameraSelector::getInstance().selectCameras(filtered);
    }

    // get the decompressors
    _decompressors = std::move(programInfo.getDecompressors());
    // assign the mesh
    this->setMesh(std::move(programInfo.getMeshLoader()));
    // assign the texture
    this->addTexture(std::move(programInfo.getTexturesLoader()));
    // get the depth shader
    _depthShader = std::move(programInfo.getDepthShader());

    if(_headless){
        // texture that will be rendered to
        glGenTextures(1, &_headlessRenderTarget);
        glBindTexture(GL_TEXTURE_2D, _headlessRenderTarget);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            _windowWidth, _windowHeight, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, NULL
        );
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _headlessRenderTarget, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "ERROR::FRAMEBUFFER::NOT_COMPLETED" << std::endl;
            exit(EXIT_FAILURE);
        }
        _headlessFrame = programInfo.headlessFrame();
    }
}


void Application::loadFrame(const size_t frame) {
    // run the decompressors
    for (auto& decompressor : _decompressors) {
        decompressor->decompress(frame, _mesh, _textureLoader);
    }

    // use the pipeline shaders
    _shaderProgram->use();

    // update what cameras are used on the GPU
    CameraSelector::getInstance().selectCameras(frame, getViewMatrix());

    if(_headlessViewpoint != -1){
        // prevent the use of the viewpoint camera when generating viewpoint
        auto cameras = CameraSelector::getInstance().getCameras();
        std::vector<int> filtered;
        std::copy_if(
            cameras.begin(), cameras.end(), std::back_inserter(filtered),
            [this](int x){return x != _headlessViewpoint;}
        );
        CameraSelector::getInstance().selectCameras(filtered);
    }

    // update the texture
    if (_textureLoader) {
      PROFILE_SCOPE("Texture load");
      _textureLoader->loadFrame(frame);
      glFinish();
    }

    CameraSelector::getInstance().setUniforms(*_shaderProgram);
    // update the mesh
    {
      PROFILE_SCOPE("Inpainting load");
      _mesh->fillEBO(frame);
      _mesh->fillVBO(frame);
      glFinish();
    }

    // compute the depth textures
    {
      PROFILE_SCOPE("Depth pass");
      _depthShader->run(_mesh);
      glFinish();
    }

    // select application shader for rendering
    _shaderProgram->use();
    _depthShader->getDepthTexture().assignToShader(*_shaderProgram);
    _depthShader->getDepthTexture().bind();
}

void Application::runHeadless(){
  loadFrame(_headlessFrame);
  _mesh->renderMesh(_textureLoader);

  std::vector<glm::uint8> pixels(_windowWidth * _windowHeight * 4);
  glReadPixels(0, 0, _windowWidth, _windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

  size_t usedCameras = CameraSelector::getInstance().getSelectedCount();
  const std::string outPath = "output.png";
  writeBufferToPNG(outPath, pixels.data(), _windowWidth, _windowHeight);
}


void Application::run() {
    if(_headless){
        runHeadless();
        return;
    }

    loadFrame(0);

    bool quit = false;
    unsigned int frame = 0;
    double lastTime = glfwGetTime();
    bool updated = true;
    double itStartTime = glfwGetTime();
    while (!glfwWindowShouldClose(_window) && !quit) {
        PROFILE_NEXT_IT();
        // Handles user input and updates the view matrix accordingly
        double elapsedTime = _paused ? 0.0f : glfwGetTime() - lastTime;
        if (_paused || elapsedTime >= (1.0/30.0)) {
            quit = handleUserInput();
            CameraSelector::getInstance().selectCameras(frame, getViewMatrix());
        }

        if (elapsedTime >= (1.0/30.0)) {
            PROFILE_NEXT_FRAME();
            PROFILE_RESET_IT();
            frame = (frame + 1) %_totalFrames; //(frame + (int)(elapsedTime / (1.0/30.0))) % totalFrames; //
            updated = false;
            lastTime = glfwGetTime();
        }

        PROFILE_SCOPE("Iteration");
        // update the frame data to new frame if needed
        if (!updated) {
            loadFrame(frame);
            // indicate that we have updated
            updated = true;
        }

        // call OpenGL to render the mesh
        {
          PROFILE_SCOPE("Render");
          _mesh->renderMesh(_textureLoader);

          // Swap the front and back buffers, so the screen texture that we just drew on becomes visible
          glfwSwapBuffers(_window);
          glFinish();
        }
        // Poll for and process events, e.g. keyboard handling
        glfwPollEvents();

        if(frame == 298){
          quit = true;
        }
        glFinish();

        // output fps
        /*std::cout << "\r" << "Frame: " << frame << "; FPS: " << 1 / (glfwGetTime() - itStartTime) << std::flush;*/
    }
}

bool Application::handleUserInput(){
    bool quit = false;
    if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(_window, true);
    }
    if (glfwGetKey(_window, GLFW_KEY_Q) == GLFW_PRESS) {
        quit = true;
    }

    glm::vec3 movement = glm::vec3(0);
    if (glfwGetKey(_window, GLFW_KEY_K) == GLFW_PRESS) { // up
        movement.y = 1;
    }
    if (glfwGetKey(_window, GLFW_KEY_J) == GLFW_PRESS) { // down
        movement.y = -1;
    }
    if (glfwGetKey(_window, GLFW_KEY_H) == GLFW_PRESS) { // left
        movement.x = 1;
    }
    if (glfwGetKey(_window, GLFW_KEY_L) == GLFW_PRESS) { // right
        movement.x = -1;
    }

    if (glfwGetKey(_window, GLFW_KEY_I) == GLFW_PRESS) { // zoom in
        movement.z = -1;
    }
    if (glfwGetKey(_window, GLFW_KEY_O) == GLFW_PRESS) { // zoom out
        movement.z = 1;
    }

    if (glfwGetKey(_window, GLFW_KEY_LEFT) == GLFW_PRESS) { // Yaw left
        _yaw += _rotationSpeed;
    }
    if (glfwGetKey(_window, GLFW_KEY_RIGHT) == GLFW_PRESS) { // Yaw right
        _yaw -= _rotationSpeed;
    }
    if (glfwGetKey(_window, GLFW_KEY_UP) == GLFW_PRESS) { // Pitch up
        _pitch -= _rotationSpeed;
    }
    if (glfwGetKey(_window, GLFW_KEY_DOWN) == GLFW_PRESS) { // Pitch down
        _pitch += _rotationSpeed;
    }


    if (glfwGetKey(_window, GLFW_KEY_P) == GLFW_PRESS) {
        glm::mat4 view = getViewMatrix();
        std::cout << "(" << std::endl;
        for(size_t row = 0; row < 4; ++row){
            for(size_t col = 0; col < 4; ++col){
                std::cout << view[row][col] << ", " << std::flush;
            }
            std::cout << std::endl;
        }
        std::cout << ")" << std::endl;
    }
    if (glfwGetKey(_window, GLFW_KEY_G) == GLFW_PRESS) { // Go
        _paused = false;
    }
    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS) { // Stop
        _paused = true;
    }


    movement *= _cameraSpeed;
    _accumMovement += movement;
    setViewMatrix();
    // if (movement != glm::vec3(0)) {
    //     // update the net movement vector
    //     // compute new location of camera and set it for the shaders
    //     setViewMatrix();
    // }

    return quit;
}

Application::~Application() {
    glfwTerminate();
}

