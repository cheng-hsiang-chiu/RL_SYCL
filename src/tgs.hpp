#pragma once

#include <iostream>
#include <atomic>
#include <vector>
#include <memory>
#include <list>
#include <utility>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>

#include "policy.hpp"

namespace tgs {

class RL_Policy;
class TGS;

struct Task {
  size_t ID;

  // the row dimension of the matrix
  size_t M;

  // the column dimension of the matrix
  size_t N;

  // atomic dependency
  std::atomic<int> dependency = 0;

  Task(const size_t id, const size_t m, const size_t n) :
    ID{id}, M{m}, N{n} {} 
};



class ThreadPool {

public:

  ThreadPool(const ThreadPool &) = delete;
  
  ThreadPool(ThreadPool &&) = delete;
  
  ThreadPool &operator=(const ThreadPool &) = delete;
  
  ThreadPool &operator=(ThreadPool &&) = delete;
  
  ThreadPool(const size_t, TGS*);

  virtual ~ThreadPool();

  template<typename T>
  void enqueue(T&&);

private:

  TGS* _tgs; 

  RL_Policy _rl;

  std::mutex _mtx;

  std::condition_variable _cv;

  std::vector<std::thread> _workers;

  std::queue<std::shared_ptr<Task>> _queue_tasks;

  template<typename T>
  void _process(T&&);
};

// task graph scheduler class  
class TGS {

friend class ThreadPool;

public:

  TGS(const size_t);

  void dump(std::ostream&) const;

  void schedule();

private:
  size_t _V;

  size_t _E;

  std::vector<std::shared_ptr<Task>> _tasks;
  
  std::vector<std::shared_ptr<Task>> _sorted_tasks;

  std::vector<std::vector<size_t>> _graph;
 
  ThreadPool _tpool; 

  void _topological_sort();  
};



// TGS construtor
inline TGS::TGS(const size_t num_threads) : _tpool(num_threads, this) {
  std::cout << num_threads << " concurrent threads are supported.\n";

  std::cin >> _V >> _E;

  _graph.resize(_V);
  _tasks.resize(_V);
  _sorted_tasks.resize(_V);

  // parse the meta data of every task
  for (size_t i = 0; i < _V; ++i) {
    size_t id, m, n;
    std::cin >> id >> m >> n;
  
    _tasks[id] = std::make_shared<Task>(id, m, n);;
  }
  
  // parse the edges
  for (size_t i = 0; i < _E; ++i) {
    size_t from, to;
    std::cin >> from >> to;

    _graph[from].push_back(to);
    ++(_tasks[to]->dependency);
  }

  _topological_sort();
}


// topological sort on _tasks
// the sorted tasks are kept in _sorted_tasks
inline void TGS::_topological_sort() {
  size_t cnt = 0;
  std::queue<size_t> q;

  std::vector<int> temp(_V);
  // push tasks of zero dependency
  for (auto& t : _tasks) {
    if (t->dependency == 0) {
      q.push(t->ID); 
    }
    temp[t->ID] = t->dependency;
  }

  while (!q.empty()) {
    size_t id = q.front();
    _sorted_tasks[cnt++] = _tasks[id];
    q.pop();

    for (size_t i = 0; i < _graph[id].size(); ++i) {
      if (--(temp[_graph[id][i]]) == 0) {
        q.push(_tasks[_graph[id][i]]->ID); 
      }
    }
  }
}


inline void TGS::schedule() {
  while (!_sorted_tasks.empty()) {
    if ((*_sorted_tasks.begin())->dependency == 0) {
      _tpool.enqueue(*_sorted_tasks.begin()); 
      _sorted_tasks.erase(_sorted_tasks.begin());
    }
  }
}








inline void TGS::dump(std::ostream& os) const {
  //for (auto& task : _tasks) {
  //  os << "Task["            << task->ID         << "]\n"
  //     << "   M : "          << task->M          << '\n'
  //     << "   N : "          << task->N          << '\n'
  //     << "   dependency : " << task->dependency << '\n';
  //}
  
  for (auto& task : _sorted_tasks) {
    os << "Sorted Task["     << task->ID         << "]\n"
       << "   M : "          << task->M          << '\n'
       << "   N : "          << task->N          << '\n'
       << "   dependency : " << task->dependency << '\n';
  }
}






// destructor
inline ThreadPool::~ThreadPool() {
  for (auto& w : _workers) {
    w.join();
  }
}


// constructor
inline ThreadPool::ThreadPool(const size_t num_threads, TGS* t) { 
  _tgs = t;
  for (size_t i = 0; i < num_threads; ++i) {
    _workers.emplace_back([&]() {
      std::shared_ptr<Task> task;
      
      while (1) {
        {
          std::unique_lock lk(_mtx);
          _cv.wait(lk, [&]() { return !_queue_tasks.empty(); });
          
          task = _queue_tasks.front();
          _queue_tasks.pop();
        }

        auto policy = _rl.policy(task);
        _process(task);
      }
    });
  }
}


template<typename T>
inline void ThreadPool::enqueue(T&& task) {
  {
    std::unique_lock lk(_mtx);

    _queue_tasks.emplace(std::forward<T>(task));
  }

  _cv.notify_one();
}


template<typename T>
inline void ThreadPool::_process(T&& task) {
  std::cout << "Thread " << std::this_thread::get_id()
            << " is processing task " << task->ID << '\n';

  for (auto& tid : _tgs->_graph[task->ID]) {
    std::cout << "tid = " << tid << '\n';
    _tgs->_sorted_tasks[tid]->dependency.fetch_sub(1, std::memory_order_acq_rel);  
  }
}
       







} // end of namespace tgs
