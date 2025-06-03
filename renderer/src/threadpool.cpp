/**
* Created by Brent Matthys on 29/03/2025
*/

#include "threadpool.h"

ThreadPool::ThreadPool(const size_t numThreads): _isRunning(true) {
  // create all the threads
  for (size_t i = 0; i < numThreads; i++) {
    _threads.emplace_back([this]() { // driver function for each thread
      while (true) {
        std::function<void()> task;
        { // critical section, only 1 thread should claim a task
          std::unique_lock<std::mutex> lock(_queueMutex); // acquire queue
          _condition.wait(lock, [this] { // only proceed if pool is stopping or task is available
              return !_isRunning || !_taskQueue.empty();
          });

          if (!_isRunning && _taskQueue.empty()) { // stop thread if pool is stopping AND no tasks available
              return;
          }

          task = std::move(_taskQueue.front()); // get the task
          _taskQueue.pop(); // remove the task from the queue
          // lock variable goes out of scope, so the _queueMutex becomes
          // free for another thread to claim
        }
        task();
      }
    });
  }
}

ThreadPool::~ThreadPool() {
    _isRunning = false;
    _condition.notify_all(); // unblock threads so they can stop
    for (auto& thread : _threads) { // join the threads back to the main thread
        if (thread.joinable()) {
            thread.join();
        }
    }
}

ThreadPool &ThreadPool::getInstance() {
    static ThreadPool instance(std::thread::hardware_concurrency());
    //static ThreadPool instance(1);
    return instance;
}
