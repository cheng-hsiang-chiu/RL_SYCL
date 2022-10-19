#pragma once

#include <thread>
#include <vector>
#include <functional>
#include <iostream>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>


namespace threadpool {




class ThreadPool {

public:

  ThreadPool(const ThreadPool &) = delete;
  
  ThreadPool(ThreadPool &&) = delete;
  
  ThreadPool &operator=(const ThreadPool &) = delete;
  
  ThreadPool &operator=(ThreadPool &&) = delete;
  
  ThreadPool(const size_t);

  virtual ~ThreadPool();

  template <class F, class... Args>
  std::future<std::result_of_t<F(Args...)>> enqueue(F&&, Args&&...);

private:

  bool _stop = false;

  std::mutex _mtx;

  std::condition_variable _cv;

  std::vector<std::thread> _workers;

  std::queue<std::function<void()>> _tasks;
};


// destructor
inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lk(_mtx);
    _stop = true;
  }

  _cv.notify_all();

  for (auto& w : _workers) {
    w.join();
  }
}


// constructor
inline ThreadPool::ThreadPool(const size_t num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    _workers.emplace_back([&]() {
      while (1) {
        std::function<void()> task;
        /* pop a task from queue, and execute it. */
        {
          std::unique_lock lk(_mtx);
          _cv.wait(lk, [&]() { return _stop || !_tasks.empty(); });
          
          if (_stop && _tasks.empty()) {
            return;
          }
          
          /* even if stop = 1, once tasks is not empty, then
           * excucte the task until tasks queue become empty
           */
          task = std::move(_tasks.front());
          _tasks.pop();
        }
        task();
      }
    });
  }
}


template<typename F, typename... Args>
inline std::future<typename std::result_of<F(Args...)>::type>
ThreadPool::enqueue(F&& f, Args&&... args) {
  
  // The return type of task F
  using rtype = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<rtype()>>(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<rtype> res = task->get_future();

  {
    std::unique_lock lk(_mtx);

    if (_stop) {
      throw std::runtime_error("Stop enqueuing new tasks.\n");
    }

    _tasks.emplace([&task](){ (*task)(); });
  }

  _cv.notify_one();

  return res;
}












} // end of namespace threadpool
