cmake_minimum_required(VERSION 3.0)

project(RTDLF)

set(CMAKE_CXX_STANDARD 17)

# Add OpenGL
find_package(OpenGL REQUIRED)
# add threads
find_package(Threads)

set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "Disable Wayland backend")
set(GLFW_BUILD_X11 ON CACHE BOOL "Enable X11 backend")

# Add Glad
add_library(glad src/glad.c)
include_directories(include)

# Add GLFW as a submodule (since it has its own CMake files)
add_subdirectory(lib/glfw)
include_directories(lib/glfw/include)

# Add GLM
include_directories(lib/glm/glm)

# Define the paths to the source files of the renderer
set(APP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_selector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/threadpool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/application.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/shader_program.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/compute/compute_shader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/depth/depth_shader.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/decompress/vcl_decompressor.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshes/mesh_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshes/color_mesh_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshes/obj_mesh_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshes/raw_mesh_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshes/dummy_mesh_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/meshes/voxel_pcl_loader.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/textures_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/codec_texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/full_img_texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/raw_imgs_texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/jpeg_texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/textures/yuv_texture.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_wrappers/atomic_counter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_wrappers/indirect_command.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/uniforms/camera_uniforms.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/input/jpeg_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/input/json_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/input/program_info.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/output/stbi_writer.cpp
)

# Define the paths to the header files of the renderer
set(APP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/threadpool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/profiler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/camera_selector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/application.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shaders/shader_program.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shaders/depth_shader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shaders/compute/compute_shader.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/decompress/decompressor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/decompress/vcl_decompressor.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshes/mesh_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshes/color_mesh_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshes/obj_mesh_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshes/raw_mesh_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshes/dummy_mesh_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/meshes/voxel_pcl_loader.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/textures_loader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/codec_texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/full_img_texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/raw_imgs_texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/jpeg_texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/textures/yuv_texture.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_wrappers/ssbo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_wrappers/atomic_counter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_wrappers/indirect_command.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/uniforms/uniforms.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/uniforms/camera_uniforms.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/input/json_reader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/input/reader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/input/program_info.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/input/buffered_loader.h

    ${CMAKE_CURRENT_SOURCE_DIR}/include/output/writer.h
)


# Define the paths to the source files of the encoder
set(ENCODER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/threadpool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/dct_matrix_provider.cpp

    # io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/io/yuv_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/io/frame_writer.cpp

    # structure
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/bit_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/frame.cpp

    # encoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/dct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/exp_golomb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/frame_transformer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/quantiser.cpp
)

# Define the paths to the header files of the encoder
set(ENCODER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/threadpool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/dct_matrix_provider.h

    # io
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/io/yuv_reader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/io/frame_writer.h

    # structure
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/structure/bit_buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/structure/frame.h

    # encoder
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/encoder/dct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/encoder/encoder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/encoder/exp_golomb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/encoder/frame_transformer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/encoder/quantiser.h
)

# Define the paths to the source files of the decoder
set(DECODER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/threadpool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/dct_matrix_provider.cpp

    # io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/io/yuv_reader.cpp

    # structure
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/frame.cpp

    # decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/decoder/decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/decoder/i_dct.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/decoder/i_exp_golomb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/decoder/i_quantiser.cpp
)

# Define the paths to the header files of the decoder
set(DECODER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/threadpool.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/dct_matrix_provider.h

    # io
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/io/yuv_reader.h

    # structure
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/structure/frame.h

    # decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/decoder/decoder.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/decoder/i_dct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/decoder/i_exp_golomb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/decoder/i_quantiser.h
)



# Define the paths to the shader files
set(APP_SHADERS
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/vertex.fs
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/geometry.fs
        # ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/fragment.fs
)

# Give them a name
source_group( "sources" FILES ${APP_SOURCES} )
source_group( "headers" FILES ${APP_HEADERS} )
source_group( "shaders" FILES ${APP_SHADERS} )

# Use the source files for the project
add_executable(${PROJECT_NAME} ${APP_SOURCES} ${APP_HEADERS} ${APP_SHADERS} ${DECODER_SOURCES} ${DECODER_HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} glfw glad ${CMAKE_DL_LIBS} Threads::Threads)

# Define the string "CMAKELISTS_SOURCE_DIR", which is used in main.cpp to more easily find the shader files
target_compile_definitions(${PROJECT_NAME} PUBLIC CMAKELISTS_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")


# ============= #
# Codec encoder #
# ============= #


add_executable(codec_encoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/main.cpp
    ${ENCODER_SOURCES}
    ${ENCODER_HEADERS}
)

# ============= #
# Codec decoder #
# ============= #

add_executable(codec_decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/decoder/main.cpp
    ${DECODER_SOURCES}
    ${DECODER_HEADERS}
)

# ===== #
# Tests #
# ===== #
set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/threadpool.cpp
    # io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/io/yuv_reader.cpp

    # structure
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/bit_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/structure/frame.cpp

    # decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/encoder/exp_golomb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/codec/decoder/i_exp_golomb.cpp
)

set(TEST_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/threadpool.h
    # io
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/io/yuv_reader.h


    # structure
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/structure/bit_buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/structure/frame.h

    # decoder
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/encoder/exp_golomb.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/codec/decoder/i_exp_golomb.h
)

set(TEST_SUITS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/tests_exp_golomb.cpp

)

add_executable(tests
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/doctest.h
    ${TEST_SUITS}
    ${TEST_SOURCES}
    ${TEST_HEADERS}
)

