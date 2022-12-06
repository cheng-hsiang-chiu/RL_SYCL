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

#include <CL/sycl.hpp>

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

  // ID of the worker that processes the task 
  int worker_id = -1;

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

  void set_MM(const size_t);

private:

  size_t _MM = 0;

  bool _stop = false;

  std::atomic<size_t> _processed = 0;

  size_t _num_threads;

  TGS* _tgs; 

  RL_Policy _rl;

  std::vector<std::mutex> _mtxs;

  std::vector<std::condition_variable> _cvs;

  std::vector<std::thread> _workers;

  std::vector<std::queue<Task*>> _queues;

  std::vector<std::vector<int>> _results;

  std::vector<sycl::queue> _sycl_gpu_queues;
  
  std::vector<sycl::queue> _sycl_cpu_queues;
  
  template<typename T>
  void _process(size_t, T&&);
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

  // maximum row of matrix
  size_t _MM = 0;

  std::vector<std::unique_ptr<Task>> _tasks;
  
  std::vector<std::vector<size_t>> _graph;
 
  ThreadPool _tpool; 
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
    
    _tasks[id] = std::make_unique<Task>(id, m, n);
    
    _MM = _MM > m ? _MM : m;
  }
  _tpool.set_MM(_MM);
  
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
  _queues(num_threads+1), _mtxs(num_threads+1), _cvs(num_threads+1),
  _results(num_threads), _rl{num_threads} { 
   
  _num_threads = num_threads;
  _tgs = t;
 
  for (size_t i = 0; i < _num_threads; ++i) {

    // every worker has its own sycl queue  
    _sycl_gpu_queues.emplace_back(sycl::queue{sycl::gpu_selector_v});
    _sycl_cpu_queues.emplace_back(sycl::queue{sycl::cpu_selector_v});

    _workers.emplace_back([&, id=i]() {
      Task* task(nullptr);
      // TODO think about the bug
      //size_t id = i;
      while (1) {
        {
          std::unique_lock<std::mutex> lk(_mtxs[id]);
          _cvs[id].wait(lk, [&]() { 
            return !_queues[id].empty() || _stop; 
          });
        
          if(_stop) {
            return;
          }
          
          if(!_queues[id].empty()) {
            task = _queues[id].front();
            _queues[id].pop();
          }
        }

        _process(id, task);
      }
    });
  }

  // master thread does the scheduling
  _workers.emplace_back([&](){
    Task* task(nullptr);
     
    while(1) {
      {
        std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
        _cvs[_num_threads].wait(lk, [&](){
          return !_queues[_num_threads].empty() || _stop;
        }); 

        if(_stop) {
          return;
        }
        
        if(!_queues[_num_threads].empty()) {
          task = _queues[_num_threads].front();
          _queues[_num_threads].pop();  
        }
      }
      
      // TODO: yile
      // just an example to output load
      double loadavg[3];
      getloadavg(loadavg, 3);
      
      printf("current system loadavg %.3lf %.3lf %.3lf\n", loadavg[0], loadavg[1], loadavg[2]);
      auto policy = _rl.policy(task);
      printf("Master decides to run task %zu with the policy: worker %ld, accelerator %d\n", task->ID, policy.first, policy.second);
      
      {
        std::unique_lock<std::mutex> lk(_mtxs[policy.first]);
        task->accelerator = policy.second;
        task->worker_id = policy.first;
        _queues[policy.first].emplace(task);
      }
      
      _cvs[policy.first].notify_one();

      // after this action
    }
  });
}


template<typename T>
inline void ThreadPool::enqueue(T&& task) {
  {
    std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
    _queues[_num_threads].emplace(std::forward<T>(task));
  }
  _cvs[_num_threads].notify_one();
}


// set the maximum row of a matrix
inline void ThreadPool::set_MM(const size_t mm) {
  _MM = mm;
  
  for (size_t i = 0; i < _num_threads; ++i) {  
    // resize and initialize every result 
    _results[i].resize(_MM * _MM, 0);
  }
}


template<typename T>
inline void ThreadPool::_process(size_t id, T&& task) {

  std::ostringstream oss;

  // TODO: add SYCL kernel based on the policy

  // offload to gpu or cpu
  sycl::queue q;

  try {
    if (task->accelerator == Accelerator::CPU) {
      q = _sycl_cpu_queues[id];
    }
    else {
      q = _sycl_gpu_queues[id];
    }
  } catch (sycl::exception const& e) {
    if (task->accelerator == Accelerator::CPU) {
      oss << "Cannot select a CPU : "
          << e.what() << '\n'
          << "Using a GPU device\n";
      q = _sycl_gpu_queues[id];
    }
    else {
      oss << "Cannot select a GPU : "
          << e.what() << '\n'
          << "Using a CPU device\n";
      q = _sycl_cpu_queues[id];
    }
    printf("%s", oss.str().c_str());
    oss.str("");
  }

  oss << "Worker "        << id 
      << " submits task " << task->ID 
      << " to " 
      << q.get_device().get_info<sycl::info::device::name>()
      << std::endl;
  printf("%s", oss.str().c_str());
  oss.str("");

  int sum = 0;
  for (auto& r : _results[id]) {
    sum += r;
  }
  
  //oss << "sum = " << sum << std::endl;
  //printf("%s", oss.str().c_str());
  //oss.str("");

  size_t M = task->M;
  size_t N = task->N;

  // declare three USM pointers to three matrixes
  // da points to matrix a, db to matrix b, dc to matrix c

  int* da = sycl::malloc_shared<int>(M*N, q);
  int* db = sycl::malloc_shared<int>(N*M, q);
  int* dc = sycl::malloc_shared<int>(M*M, q);

  // initialize matrix a
  q.parallel_for(
    sycl::range<1>(M*N),
    [=](sycl::id<1> i) {
      da[i] = sum - id;
    }
  );

  // initialize matrix b
  q.parallel_for(
    sycl::range<1>(N*M),
    [=](sycl::id<1> i) {
      db[i] = sum + id;
    }
  ).wait();

  auto _M = (M % 16 == 0) ? M : (M + 16 - M % 16);

  // matrix multiplication c = a * b
  q.parallel_for(
    sycl::nd_range<2>{sycl::range<2>(_M, _M), sycl::range<2>(16, 16)},
    [=](sycl::nd_item<2> item) {
    
      int row = item.get_global_id(0);
      int col = item.get_global_id(1);
      
      if(row < M && col < M) {
        int sum = 0;
        
        for(int n = 0; n < N; n++) {
            sum += da[row * N + n] * db[n * M + col];
        }
        dc[row * M + col] = sum;
      }
    }
  ).wait();


  // copy result back to worker's local memory
  for (size_t i = 0; i < M*M; i++) {
    _results[id][i%(_MM*_MM)] = *(dc+i);
  }


  // decrement the dependencies
  for (auto& tid : _tgs->_graph[task->ID]) {

    if(_tgs->_tasks[tid]->join_counter.fetch_sub(1)==1){
      enqueue(_tgs->_tasks[tid].get());
    }  
  }

  if (_processed.fetch_add(1) + 1 == _tgs->_V) {
    _stop = true;
    for(auto& cv : _cvs) {
      cv.notify_one();
    }
  }
}




} // end of namespace tgs
