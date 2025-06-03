//
// Created by brent on 12/11/24.
//

#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <glm.hpp>

/**
 * Reads a binary file to buffer in the given type.
 * This function creates a buffer in memory, the
 * caller is responsible for freeing this buffer.
 * @tparam T The buffer type
 * @param filePath The path to load the buffer from
 * @param offset The amount of elements to skip before reading
 * @param bufferSize The size of the loaded buffer. if set, used, if 0, this will be set to #elements in the file
 * @return A pointer to the buffer
 */
template<typename T>
T* readBuffer(const std::string& filePath, glm::uint64 offset, glm::uint32& bufferSize) {
  static_assert(std::is_trivially_copyable<T>::value, "ERROR::RAW::T_MUST_BE_TRIVIALLY_COPYABLE");

  // Open the binary file
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "ERROR::RAW::FAILED_TO_OPEN: " + filePath;
    exit(EXIT_FAILURE);
  }

  if(bufferSize == 0){
    //Get the file size and validate
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize % sizeof(T) != 0) {
      std::cerr << "ERROR::RAW::FILE_SIZE_NO_MULTIPLE_OF_ELEMENT_SIZE: " + filePath;
      exit(EXIT_FAILURE);
    }
    // Read the file into a buffer
    bufferSize = fileSize / sizeof(T);
  }

  T* buffer = new T[bufferSize];
  file.seekg(static_cast<std::streamoff>(offset * sizeof(T)), std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(buffer), bufferSize*sizeof(T))) {
      std::cerr << "ERROR::RAW::FAILED_TO_READ: " + filePath;
      delete[] buffer;
      exit(EXIT_FAILURE);
  }

  return buffer;
}

/**
 * Reads a binary file to buffer in the given type.
 * This function creates a buffer in memory, the
 * caller is responsible for freeing this buffer.
 * @tparam T The buffer type
 * @param filePath The path to load the buffer from
 * @param bufferSize The size of the loaded buffer (output param)
 * @return A pointer to the buffer
 */
template<typename T>
T* readRawFileToBuffer(const std::string& filePath, glm::uint32& bufferSize) {
  bufferSize = 0;
  return readBuffer<T>(filePath, 0, bufferSize);
}

/**
 * Reads a jpeg file to buffer.
 * This function creates a buffer in memory, the
 * caller is responsible for freeing this buffer.
 * @param filePath The path to load the buffer from
 * @param bufferSize The size of the loaded buffer (output param)
 * @return A pointer to the buffer
 */
glm::uint8* readJPEGToBuffer(const std::string& filePath, glm::uint32& bufferSize); 

//#endif //READER_H
