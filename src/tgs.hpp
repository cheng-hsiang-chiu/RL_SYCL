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
#include <chrono>
#include <fstream>

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

  std::vector<std::deque<Task*>> _queues;

  template<typename T>
  void _process(size_t, T&&);

  void _state_query() ;
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
  
  std::vector<std::vector<size_t>> _graph;
 
  std::vector<std::chrono::high_resolution_clock::time_point> _timestamp;
  
  ThreadPool _tpool; 

  void _dump_timestamp() const;
};



// TGS construtor
inline TGS::TGS(const size_t num_threads) : _tpool(num_threads, this) {
  std::cout << num_threads << " concurrent threads are supported.\n";

  std::cin >> _V >> _E;

  _graph.resize(_V);
  _tasks.resize(_V);
  _timestamp.resize(_V);
  
  // parse the meta data of every task
  for (size_t i = 0; i < _V; ++i) {
    size_t id, m, n;
    std::cin >> id >> m >> n;
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
  _queues(num_threads+1), _mtxs(num_threads+1), _cvs(num_threads+1), _rl{num_threads} { 
   
  _num_threads = num_threads;
  _tgs = t;
  
  for (size_t i = 0; i < _num_threads; ++i) {
    
    // definitions of the worker 
    _workers.emplace_back([&, id=i]() {
      Task* task(nullptr);
      while (1) {
        {
          std::unique_lock<std::mutex> lk(_mtxs[id]);
          _cvs[id].wait(lk, [&]() { 
            return !_queues[id].empty() || stop; 
          });
       
          // the scheduler has scheduled all tasks 
          if(stop) {
            return;
          }
          
          // worker i has tasks in its queue
          // and pops a task from the queue
          if(!_queues[id].empty()) {
            task = _queues[id].front();
            _queues[id].pop_front();
          }
        }

        // start to process the task
        _process(id, task);
      }
    });
  }

  // master thread does the scheduling
  _workers.emplace_back([&](){
    Task* task(nullptr);
    size_t idx = 0;
     
    while(1) {
      {
        std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
        _cvs[_num_threads].wait(lk, [&](){
          return !_queues[_num_threads].empty() || stop;
        }); 

        // the scheduler has scheduled all tasks
        if(stop) {
          return;
        }
       
        // master has tasks in its queue
        // and pops a task from its queue 
        if(!_queues[_num_threads].empty()) {
          task = _queues[_num_threads].front();
          _queues[_num_threads].pop_front();  
        }
      }
      
      // record the timstamp before master calls RL for an action recommendation
      // timestamp is used for plotting histogram only
      // could comment the line if not necessary
      _tgs->_timestamp[idx++] = std::chrono::high_resolution_clock::now();

      // use _state_query to query the state information  
      //_state_query();
      
      // master begins to call RL for an action regarding the task
      auto policy = _rl.policy(task);
      //printf("Master decides to run task %zu with the policy:worker %ld, accelerator %d\n", 
      //        task->ID, policy.first, policy.second);
      
      // master gets the action recommendation and pushes the task to
      // the corresponding worker's queue
      {
        std::unique_lock<std::mutex> lk(_mtxs[policy.first]);
        task->accelerator = policy.second;
        _queues[policy.first].emplace_back(task);
      }
  
      // use _state_query to query the state information  
      //_state_query();
      
      _cvs[policy.first].notify_one();
    }
  });
}


// enqueue a task in master's queue
template<typename T>
inline void ThreadPool::enqueue(T&& task) {
  // acquire the lock of master thread
  // before push the task into master's queue
  {
    std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
    _queues[_num_threads].emplace_back(std::forward<T>(task));
  }
  _cvs[_num_threads].notify_one();
}


template<typename T>
inline void ThreadPool::_process(size_t id, T&& task) {

  std::ostringstream oss;
  oss << "Worker " << id << " is processing task " << task->ID << std::endl;
  printf("%s\n", oss.str().c_str());

  // TODO: add SYCL kernel based on the policy
  // right now we use sleep to simulate the loading of a task
  std::this_thread::sleep_for(std::chrono::milliseconds(task->M * task->N));

  // decrement the dependencies
  for (auto& tid : _tgs->_graph[task->ID]) {
    if(_tgs->_tasks[tid]->join_counter.fetch_sub(1)==1){
      enqueue(_tgs->_tasks[tid].get());
    }  
  }

  // finished processing all tasks
  // stop the scheduler
  if (_processed.fetch_add(1) + 1 == _tgs->_V) {
    stop = true;
    for(auto& cv : _cvs) {
      cv.notify_one();
    }

    // dump the timestamp 
    // used for plotting histogram only
    _tgs->_dump_timestamp();
  }
}


// query the state information
inline void ThreadPool::_state_query() {
  std::mutex io_mutex;
  {
    std::unique_lock<std::mutex> lk(io_mutex);
    double loadavg[3];
    getloadavg(loadavg, 3);
    std::cout << "--------------------------\n"
              << "Current system loadavg : " 
              << loadavg[0] << ", "
              << loadavg[1] << ", "
              << loadavg[2] << '\n';
   
    // the followings are the information of worker's queue
    // together with the info of a task in the queue   
    for (size_t i = 0; i < _num_threads; ++i) {
      {
        std::unique_lock<std::mutex> lk(_mtxs[i]);
        std::cout << "Thread " << i << " has " 
                  << _queues[i].size() << " tasks in its queue\n";
        size_t cnt = 0;
        for (auto& t : _queues[i]) {
          std::cout << "   Queue[" << cnt++ << "] : "
                    << "Task ID " << t->ID
                    << ", M " << t->M
                    << ", N " << t->N
                    << ", accelerator " << t->accelerator
                    << '\n'; 
        }
      }
    } 
    std::cout << "--------------------------\n";
  } 
}

// dump the timestamp
// used for plotting histogram only
inline void TGS::_dump_timestamp() const {
  std::ofstream MyFile("../python_visualization/timestamp.csv");
  for (size_t i = 1; i < _timestamp.size(); ++i) {
    MyFile << std::chrono::duration_cast<std::chrono::nanoseconds>(
      _timestamp[i]-_timestamp[i-1]).count()
           << "\n";
  }

  MyFile.close();
}




} // end of namespace tgs
