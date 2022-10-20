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
#include <string>
#include <sstream>

//#include <CL/sycl.hpp>

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

  // atomic join_counter
  std::atomic<int> join_counter = 0;

  // if scheduled
  bool scheduled = false;

  // Task executed on accelerator 
  Accelerator accelerator;

  Task(const size_t id, const size_t m, const size_t n) :
    ID{id}, M{m}, N{n} {} 
};

class ThreadPool {

public:

  bool stop = false;
  
  ThreadPool(const ThreadPool &) = delete;
  
  ThreadPool(ThreadPool &&) = delete;
  
  ThreadPool &operator=(const ThreadPool &) = delete;
  
  ThreadPool &operator=(ThreadPool &&) = delete;
  
  ThreadPool(const size_t, TGS*);

  virtual ~ThreadPool();

  template<typename T>
  void enqueue(T&&);

private:

  std::atomic<size_t> _processed = 0;

  size_t _num_threads;

  TGS* _tgs; 

  RL_Policy _rl;

  std::vector<std::mutex> _mtxs;

  std::vector<std::condition_variable> _cvs;

  std::vector<std::thread> _workers;

  std::vector<std::queue<Task*>> _queues;

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

  std::vector<std::unique_ptr<Task>> _tasks;
  
  //std::vector<std::shared_ptr<Task>> _sorted_tasks;

  std::vector<std::vector<size_t>> _graph;
 
  ThreadPool _tpool; 

  //void _topological_sort();  
};



// TGS construtor
inline TGS::TGS(const size_t num_threads) : _tpool(num_threads, this) {
  std::cout << num_threads << " concurrent threads are supported.\n";

  std::cin >> _V >> _E;

  _graph.resize(_V);
  _tasks.resize(_V);

  // parse the meta data of every task
  for (size_t i = 0; i < _V; ++i) {
    size_t id, m, n;
    std::cin >> id >> m >> n;
    //std::cout << id << ' ' << m << ' ' << n << '\n'; 
    _tasks[id] = std::make_unique<Task>(id, m, n);
  }
  
  // parse the edges
  for (size_t i = 0; i < _E; ++i) {
    size_t from, to;
    std::cin >> from >> to;

    _graph[from].push_back(to);

    _tasks[to]->join_counter.fetch_add(1, std::memory_order_relaxed);
  }
}


inline void TGS::schedule() {

  std::vector<Task*> source;

  for(const auto& t: _tasks) {
    if(t->join_counter.load() == 0) {
      source.push_back(t.get());
    }
  }

  for(auto task : source){
    _tpool.enqueue(task);
  }
}



inline void TGS::dump(std::ostream& os) const {
  for (auto& task : _tasks) {
    os << "Task["              << task->ID           << "]\n"
       << "   M : "            << task->M            << '\n'
       << "   N : "            << task->N            << '\n'
       << "   join_counter : " << task->join_counter << '\n';
  }
}

// destructor
inline ThreadPool::~ThreadPool() {
  for (auto& w : _workers) {
    w.join();
  }
}


// constructor
inline ThreadPool::ThreadPool(const size_t num_threads, TGS* t) :
  _queues(num_threads+1), _mtxs(num_threads+1), _cvs(num_threads+1) { 
   
  _num_threads = num_threads;
  _tgs = t;
  
  for (size_t i = 0; i < _num_threads; ++i) {
    
    _workers.emplace_back([&, id=i]() {
      Task* task(nullptr);
      // TODO think about the bug
      //size_t id = i;
      while (1) {
        {
          //printf("worker %zu before cv\n", id);
          std::unique_lock<std::mutex> lk(_mtxs[id]);
          _cvs[id].wait(lk, [&]() { 
            return !_queues[id].empty() || stop; 
          });
          //printf("worker %zu after cv\n", id);
        
          if(stop) {
            return;
          }
          
          if(!_queues[id].empty()) {
            task = _queues[id].front();
            _queues[id].pop();
          }
        }

        _process(task);
      }
    });
  }

  // master thread does the scheduling
  _workers.emplace_back([&](){
    Task* task(nullptr);
     
    while(1) {
      {
        //printf("master thread before cv\n");
        std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
        _cvs[_num_threads].wait(lk, [&](){
          return !_queues[_num_threads].empty() || stop;
        }); 
        //printf("master thread after cv\n");

        if(stop) {
          return;
        }
        
        if(!_queues[_num_threads].empty()) {
          task = _queues[_num_threads].front();
          _queues[_num_threads].pop();  
        }
      }
      
      auto policy = _rl.policy(task);
      printf("task %zu has policy[worker %ld, accelerator %d]\n", task->ID, policy.first, policy.second);
      
      {
        std::unique_lock<std::mutex> lk(_mtxs[policy.first]);
        //printf("get %zu's lock\n", policy.first);
        task->accelerator = policy.second;
        _queues[policy.first].emplace(task);
      }
      //printf("release %zu's lock\n", policy.first);
      
      _cvs[policy.first].notify_one();
    }
  });
}


template<typename T>
inline void ThreadPool::enqueue(T&& task) {
  {
    std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
    _queues[_num_threads].emplace(std::forward<T>(task));
  }
  //printf("finish enqueue task\n");
  _cvs[_num_threads].notify_one();
}


template<typename T>
inline void ThreadPool::_process(T&& task) {

  std::ostringstream oss;
  oss << "Thread " << std::this_thread::get_id() 
      << " is processing task " << task->ID << std::endl;
  printf("%s\n", oss.str().c_str());

  // TODO: add SYCL kernel based on the policy

  // decrement the dependencies
  for (auto& tid : _tgs->_graph[task->ID]) {
    //std::cout << "tid = " << tid << '\n';
    if(_tgs->_tasks[tid]->join_counter.fetch_sub(1)==1){
      enqueue(_tgs->_tasks[tid].get());
    }  
  }

  if (_processed.fetch_add(1) + 1 == _tgs->_V) {
    stop = true;
    for(auto& cv : _cvs) {
      cv.notify_one();
    }
  }
}




} // end of namespace tgs
