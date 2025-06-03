//
// Created by brent on 11/20/24.
//

#ifndef APPLICATION_H
#define APPLICATION_H

#include "shaders/depth_shader.h"
#include "shaders/shader_program.h"
#include "input/program_info.h"
#include "meshes/mesh_loader.h"
#include "textures/textures_loader.h"
#include "uniforms/uniforms.h"

#include "GLFW/glfw3.h"
#include <cstddef>
#include <glm.hpp>
#include <memory>

class Application {
private:
    GLFWwindow* _window = nullptr;

    size_t _windowWidth;
    size_t _windowHeight;

    std::unique_ptr<ShaderProgram> _shaderProgram = nullptr;
    std::unique_ptr<DepthShader> _depthShader = nullptr;
    std::vector<std::unique_ptr<Decompressor>> _decompressors;
    std::unique_ptr<MeshLoader> _mesh = nullptr;
    std::unique_ptr<TexturesLoader> _textureLoader = nullptr;

    size_t _totalFrames;
    bool _paused = false;


    float _cameraSpeed = 0.002f;
    glm::vec3 _accumMovement = glm::vec3(0);

    float _rotationSpeed = 0.02f;
    float _yaw = 0.0f; // left-right rotation
    float _pitch = 0.0f; // up-down rotation

    glm::mat4 _basicView = glm::mat4(1.0f);


    bool _headless = false;
    size_t _headlessFrame = 0;
    std::string _headlessOutPath;
    GLuint _headlessRenderTarget = 0;


    /**
     * Handle the user input.
     * @return Weather or not to quit the program.
     */
    bool handleUserInput();


    /**
     * Function that initializes the window of the application.
     * The window should be inited before any other part of the application.
     * The window can only be inited once.
     *
     * @param width The width of the window to create
     * @param height The height of the window to create
     */
    void initWindow();

    /**
     * Function that initialized the shader program of the application.
     * Should only be called once this application has a window.
     *
     * @param shaders The shaderinfo of the shaders to compile into the program.
     */
    void initShaderProgram(const std::vector<ShaderInfo>& shaders);

    /**
     * Function that sets the mesh loader.
     * Should only be called once the shader program is created.
     *
     * @param meshLoader The mesh to render.
     */
    void setMesh(std::unique_ptr<MeshLoader> meshLoader);

    /**
     * Function that adds a texture to the application.
     * Should only be called once a mesh has been set.
     *
     * @param textureLoader The texture to add.
     */
    void addTexture(std::unique_ptr<TexturesLoader> textureLoader);


    /**
     * Function that sets the uniforms for the shaders at
     * the start of the program.
     * Should only be called once the shader program is created.
     *
     * @param uniforms The uniforms object to set.
     */
    void setUniforms(const std::unique_ptr<Uniforms>& uniforms);

    /**
     * Compute the view matrix based on the basicview, the translation and rotation of the camera.
     */
    glm::mat4 getViewMatrix();

    /**
     * Set the view matrix on the GPU based on the getViewMatrix function.
     */
    void setViewMatrix();

    /**
     * Decompress the frame, and load the data to the required buffers.
     * This does not render the frame.
     * @param frame The frame to load.
     */
    void loadFrame(size_t frame);

    void runHeadless();

public:

    /**
     * Initializes this application. This is equivalent to calling
     * initWindow, initShaderProgram, setMesh, addTexture and setUniforms in a row.
     *
     * @param programInfo The info to init this app with.
     */
    void initialize(const ProgramInfo& programInfo);

    /**
     * Run the application until stopped by the user
     * This function should only be called if the shaders
     * the mesh and the textures have been set.
     */
    void run();

    ~Application();
};

#endif //APPLICATION_H
