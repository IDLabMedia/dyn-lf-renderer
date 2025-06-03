#pragma once

#include <chrono>
#include <fstream>
#include <string>
#include <mutex>
#include <nlohmann/json.hpp>

// Toggle this macro to enable/disable profiling
#define ENABLE_PROFILING 1

#if ENABLE_PROFILING

class Profiler {
private:

  std::mutex _mutex;
  std::string _outFile;
  nlohmann::json _data;
  size_t _iteration = 0;
  size_t _frame = 0;

public:
  static Profiler& get() {
    static Profiler instance;
    return instance;
  }

  void beginSession(const std::string& filepath = "profile.json") {
    std::lock_guard<std::mutex> lock(_mutex);
    _data = nlohmann::json::array();
    _outFile = filepath;
  }

  void endSession() {
    std::lock_guard<std::mutex> lock(_mutex);
    std::ofstream file(_outFile);
    file << _data.dump(4); // pretty-print with indent of 4
    file.close();
    _data.clear();
  }

  void nextIteration() {
    _iteration++;
  }

  void resetIteration() {
    _iteration = 0;
  }

  void nextFrame() {
    _frame++;
  }


  void writeProfile(const std::string& name, long long start, long long end) {
    std::lock_guard<std::mutex> lock(_mutex);
    _data.push_back({
      {"name", name},
      {"frame", _frame},
      {"iteration", _iteration},
      {"start", start},
      {"end", end},
      {"duration", end - start}
    });
  }

};

class ProfileTimer {
private:
  std::string _name;
  std::chrono::high_resolution_clock::time_point _start;
public:
  ProfileTimer(const std::string& name)
    : _name(name), _start(std::chrono::high_resolution_clock::now()) {}

  ~ProfileTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    long long startTime = std::chrono::time_point_cast<std::chrono::microseconds>(_start).time_since_epoch().count();
    long long endTime = std::chrono::time_point_cast<std::chrono::microseconds>(end).time_since_epoch().count();
    Profiler::get().writeProfile(_name, startTime, endTime);
  }

};

#define PROFILE_SCOPE(name) ProfileTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#define PROFILE_NEXT_IT() Profiler::get().nextIteration()
#define PROFILE_RESET_IT() Profiler::get().resetIteration()
#define PROFILE_NEXT_FRAME() Profiler::get().nextFrame()

#else

#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#define PROFILE_NEXT_IT()
#define PROFILE_RESET_IT()
#define PROFILE_NEXT_FRAME()

#endif
