//
// Created by brent on 12/17/24.
//

#ifndef BUFFERED_LOADER_H
#define BUFFERED_LOADER_H

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "fwd.hpp"
#include "reader.h"
#include "threadpool.h"

enum class BufferState {
    LOADED,
    LOADING,
    NOT_LOADED,
};

/**
 * Class that can load a given amount of buffers.
 * This class will load those buffers in parallel.
 * The class functions should only be called from a single thread.
 * There is no need to call these functions from different threads,
 * since this class will handle parallel compute for you.
 * @tparam T The data type of the buffers
 */
template <typename T>
class BufferedLoader {
private:
    ThreadPool& _pool = ThreadPool::getInstance();

    const unsigned int _maxBuffers;
    std::function<T*(const std::string&, glm::uint32&)> _loadFunction;

    std::vector<BufferState> _bufferStates;
    std::vector<glm::uint32> _bufferSizes;
    T** _buffers;

public:
    /**
     * Create a buffered loader.
     * Buffers will be loaded from path: prefix + index + postfix
     * @param maxBuffers The max amount of buffers to store
     */
    BufferedLoader(
    const unsigned int maxBuffers,
    std::function<T*(const std::string&, glm::uint32&)> loadFunction
    ) : _maxBuffers(maxBuffers), _loadFunction(loadFunction) {
        _bufferStates = std::vector<BufferState>(maxBuffers, BufferState::NOT_LOADED);
        _bufferSizes = std::vector<glm::uint32>(maxBuffers, 0);
        _buffers = new T*[maxBuffers];
        for (unsigned int i = 0; i < maxBuffers; i++) {
            _buffers[i] = nullptr;
        }
    };

    // destructor
    ~BufferedLoader() {
        for (unsigned int i = 0; i < _maxBuffers; i++) {
            delete[] _buffers[i];
        }
        delete[] _buffers;
    }

    /**
     * Get a preloaded buffer, or load the buffer if not loaded.
     * @param index The index of the buffer to get
     * @param path The path to the file to get the buffer from (needed if buffer isn't preloaded)
     * @return Pointer to the buffer
     */
    T* getBuffer(const unsigned int index, const std::string& path) {
        loadBuffer(index, path); // load if not loaded or loading
        while (_bufferStates[index] != BufferState::LOADED){} // wait for the buffer to be loaded
        return _buffers[index]; // if loaded nothing will be altering the buffer, so we can return it
    }

    /**
     * Get the size of a given buffer. This function will not
     * block if the buffer is still loading. You can only safely use
     * this function after a call of getBuffer or clearBuffer, but before
     * a call to loadBuffer.
     * @param index The buffer to get the size for
     * @return The size of the buffer
     */
    glm::uint32 getBufferSize(const unsigned int index) const {
        return _bufferSizes[index];
    }

    /**
     * Load a buffer from file in parallel.
     * @param index Load a buffer in parallel
     * @param path The path to the file to load to buffer
     * @return true if the loading started, false if the buffer is already loaded or loading
     */
    bool loadBuffer(const unsigned int index, const std::string& path) {
        if (_bufferStates[index] != BufferState::NOT_LOADED) return false; // buffer is being loaded or loaded
        _bufferStates[index] = BufferState::LOADING; // no need for mutex, since only this thread will be setting if NOT_LOADED
        // By setting to LOADING, we prevent other threads from starting to load this buffer
        _pool.enqueueTask([index, path, this]() {
            this->_buffers[index] = _loadFunction(path, _bufferSizes[index]); // load the buffer
            this->_bufferStates[index] = BufferState::LOADED; // indicate the buffer is loaded
        });
        return true;
    }

    /**
     * Load numBuffers buffers, starting with buffer index
     * @param index The first buffer to load
     * @param numBuffers The following buffers to load
     * @param pathGenerator Function that takes an int (this will be the index of the buffer to load) and returns a path to the file to load into the buffer
     */
    void loadBuffers(const unsigned int index, const unsigned int numBuffers, const std::function<std::string(unsigned int)>& pathGenerator) {
        for (unsigned int i = index; i < (index + numBuffers) % _maxBuffers; i = (i + 1) % _maxBuffers) {
            loadBuffer(i, pathGenerator(i));
        }
    }

    /**
     * Free the buffer at the given index
     * @param index The buffer to clear
     */
    void clearBuffer(const unsigned int index) {
        if (_bufferStates[index] == BufferState::NOT_LOADED) return; // nothing to clear
        while (_bufferStates[index] != BufferState::LOADED){} // wait for the buffer to be loaded
        // buffer is loaded here, so we free it
        _bufferStates[index] = BufferState::NOT_LOADED;
        delete[] _buffers[index];
        _buffers[index] = nullptr;
        _bufferSizes[index] = 0;
    }

    void printLoadedBuffers() const {
        for (unsigned int i = 0; i < _maxBuffers; i++) {
            std::cout << (_bufferStates[i] == BufferState::LOADED ? "L" : _bufferStates[i] == BufferState::LOADING ? "*": ".");
        }
        std::cout << std::endl;
    }
};

#endif //BUFFERED_LOADER_H
