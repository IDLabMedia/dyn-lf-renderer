/**
* Created by Brent Matthys on 29/03/2025
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <future>

class ThreadPool {
private:
  std::vector<std::thread> _threads;
  std::queue<std::function<void()>> _taskQueue;
  std::mutex _queueMutex;
  std::condition_variable _condition;
  std::atomic<bool> _isRunning;

  // Constructor and destructor are private for singleton purposes
  explicit ThreadPool(size_t numThreads);
  ~ThreadPool();

public:
  // Delete copy constructor and assignment operator to enforce singleton
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  /**
   * Get the singleton Threadpool.
   * @return The singleton Threadpool.
   */
  static ThreadPool& getInstance();

  /**
   * Enqueue a task to be executed in parallel. Note that the
   * threadpool will not lock the used resources of the task.
   * It is the job of the task to ensure thread safety.
   * The threadpool will only ensure that only 1 thread will execute
   * the task.
   */
  template <typename F, typename... Args>
  auto enqueueTask(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    // get type that our callable function f will return
    using ReturnType = std::invoke_result_t<F, Args...>;

    // create a task where we can get the result from as
    // a future later.
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    // get the future (this will later contain the result)
    std::future<ReturnType> future = task->get_future();

    { // critical section, we can only add to the queue if no thread is popping from it
      std::unique_lock<std::mutex> lock(_queueMutex);
      _taskQueue.emplace([task]() { (*task)(); });
    }

    _condition.notify_one(); // Notify that a task is ready to be executed 
    return future;
  }
};
