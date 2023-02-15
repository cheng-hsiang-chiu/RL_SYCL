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
#include <future>
//#include <omp.h>

#include <CL/sycl.hpp>

#include "policy.hpp"
#include "gpu_info.hpp"

namespace tgs {

class RL_Policy;
class TGS;

enum Weights {                                                                                                                                
  Default  = -1,
  CPUtoCPU = 0,
  GPUtoGPU = 0,
  CPUtoGPU = 2,
  GPUtoCPU = 2
};
//
//
//struct Edge {
//  // ID of the "From" task
//  size_t From_task_id;
//
//  // ID of the "To" task
//  size_t To_task_id;
//
//  // weight of the edge, initialize to Default(-1)
//  Weights weight = Weights::Default;
//};



struct Task {
  size_t ID;

  // the row dimension of the matrix
  size_t M;

  // the column dimension of the matrix
  size_t N;

  // atomic join_counter
  std::atomic<int> join_counter = 0;

  // atomic number of child
  std::atomic<int> num_child = 0; 

  // if scheduled
  bool scheduled = false;

  // Task executed on accelerator 
  Accelerator accelerator;

  // ID of the worker that processes the task 
  int worker_id = -1;

  // IDs of the parent task
  std::vector<Task*> parents;

  // pointer to a dynamically allocated matrix
  int* ptr_matrix = nullptr;

  // sycl queue 
  sycl::queue sycl_queue;

  Task(const size_t id, const size_t m, const size_t n) :
    ID{id}, M{m}, N{n} {} 
};

class ThreadPool {

public:

  ThreadPool(const ThreadPool &) = delete;
  
  ThreadPool(ThreadPool &&) = delete;
  
  ThreadPool &operator=(const ThreadPool &) = delete;
  
  ThreadPool &operator=(ThreadPool &&) = delete;
  
  ThreadPool(const size_t, const size_t, TGS*);

  virtual ~ThreadPool();

  template<typename T>
  void enqueue(T&&);

private:

  bool _stop = false;

  std::atomic<size_t> _processed = 0;

  size_t _num_threads;
  
  size_t _thread_task;

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

  TGS(const size_t, const size_t);

  void dump(std::ostream&) const;
  
  void dump_scheduling(std::ostream&);

  void schedule();

  ~TGS();  

private:

  std::vector<std::future<void>> _future;
  
  std::vector<std::promise<void>> _promise;

  size_t _V;

  size_t _E;

  std::vector<std::unique_ptr<Task>> _tasks;

  //std::vector<std::unique_ptr<Edge>> _edges;
  
  std::vector<std::vector<size_t>> _graph;
 
  ThreadPool _tpool; 
};



// TGS construtor
inline TGS::TGS(const size_t num_threads, const size_t thread_task) : 
  _promise(num_threads+1), _tpool(num_threads, thread_task, this) {
  
  std::cout << num_threads << " concurrent threads are supported.\n";

  std::cin >> _V >> _E;

  _graph.resize(_V);
  _tasks.resize(_V);
  //_edges.resize(_E);

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
    _tasks[from]->num_child.fetch_add(1, std::memory_order_relaxed);
    _tasks[to]->parents.push_back(_tasks[from].get());
  }
}


inline TGS::~TGS() {
  //for (auto& t : _tasks) {
  //  if (t->ptr_matrix && t->accelerator == Accelerator::GPU) {
  //    sycl::free(t->ptr_matrix, t->sycl_queue);
  //  }
  //  
  //  else if (t->ptr_matrix && t->accelerator == Accelerator::CPU) {
  //    delete t->ptr_matrix;
  //  }
  //}
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


/*
 * dump schedulng results
 * The first line is the number of tasks (_V)
 * The following _V lines are the scheudling results of each task
 * The following _E lines are the weights of each edge
 */
inline void TGS::dump_scheduling(std::ostream& os) {
  // wait here until all workers are done
  for (auto& fu : _future) {
    fu.get();
  }
  
  os << _V << '\n'; 

  for (auto& task : _tasks) {
    os << task->ID        << ' '
       << task->worker_id << ' '	
       << task->accelerator << '\n';
  }

  for (auto& task : _tasks) {  
    for (auto& p : task->parents) {
      switch (p->accelerator)
      {
        case Accelerator::GPU:
          switch (task->accelerator)
          {
            case Accelerator::GPU:
              // from gpu to gpu
              os << p->ID    << ' ' 
                 << task->ID << ' '
                 << "0\n";
            break;

            case Accelerator::CPU:
              // from gpu to cpu
              os << p->ID    << ' ' 
                 << task->ID << ' '
                 << "2\n";
            break;
          }
        break;

        case Accelerator::CPU:		
          switch (task->accelerator)
          {
            case Accelerator::GPU:
              // from cput to gpu
              os << p->ID    << ' ' 
                 << task->ID << ' '
                 << "2\n";
            break;

	    case Accelerator::CPU:
	      // from cpu to cpu 
              os << p->ID    << ' ' 
                 << task->ID << ' '
                 << "0\n";
            break;
          }
        break;
      }
    }
  }
}


// destructor
inline ThreadPool::~ThreadPool() {
  for (auto& w : _workers) {
    w.join();
  }
}


// constructor
inline ThreadPool::ThreadPool(const size_t num_threads, const size_t thread_task, TGS* t) :
  _queues(num_threads+1), _mtxs(num_threads+1), _cvs(num_threads+1),
  _rl{num_threads} { 
  
  _num_threads = num_threads;
  _tgs = t;
  _thread_task = thread_task; 

  for (size_t i = 0; i < _num_threads; ++i) {
    _tgs->_future.emplace_back(_tgs->_promise[i].get_future());

    // every worker has its own sycl queue  
    _sycl_gpu_queues.emplace_back(sycl::queue{sycl::gpu_selector_v});
    _sycl_cpu_queues.emplace_back(sycl::queue{sycl::cpu_selector_v});

    _workers.emplace_back([&, id=i, &p=_tgs->_promise]() {
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
            p[id].set_value();   
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
  _tgs->_future.emplace_back(_tgs->_promise[_num_threads].get_future());
  _workers.emplace_back([&, &p=_tgs->_promise](){
    Task* task(nullptr);
     
    while(1) {
      {
        std::unique_lock<std::mutex> lk(_mtxs[_num_threads]);
        _cvs[_num_threads].wait(lk, [&](){
          return !_queues[_num_threads].empty() || _stop;
        }); 

        if(_stop) {
          p[_num_threads].set_value();
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
      printf("GPU Memory Usage : %ld\n", getGPUMemUsage(0)); 
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


template<typename T>
inline void ThreadPool::_process(size_t id, T&& task) {

  std::ostringstream oss;

  // sum up all parents' sum
  int parent_sum = 0;
  int* ptr = nullptr;
  
  for (auto& p : task->parents) {
    for (int i = 0; i < p->M * p->M; ++i) {
      parent_sum += (p->ptr_matrix)[i];
    }
    
    // free parent's memory allocated on CPU or GPU
    // if i am the last child to fetch parent's matrix 
    if(p->num_child.fetch_sub(1) == 1) {
      if (p->accelerator == Accelerator::CPU) {
        delete p->ptr_matrix;
      }
      else {
	      sycl::free(p->ptr_matrix, p->sycl_queue);
      }
    }
  }
  
  /* 
  oss << "Worker "        << id 
      << " submits task " << task->ID 
      << " to ";
  if (task->accelerator == Accelerator::CPU) {
    task->sycl_queue = _sycl_cpu_queues[id];
  }
  else {
    task->sycl_queue = _sycl_gpu_queues[id];
  }
  oss << (task->sycl_queue).get_device().template get_info<sycl::info::device::name>()
      << std::endl;
  printf("%s", oss.str().c_str());
  oss.str("");
  */

  
  size_t M = task->M;
  size_t N = task->N;

  // CPU or GPU matrix multiplication
  if (task->accelerator == Accelerator::CPU) {
    //task->sycl_queue = _sycl_cpu_queues[id];
    
    // CPU version matrix multiplication
    std::vector<int> da(M*N, parent_sum - id);
    std::vector<int> db(N*M, parent_sum + id);
    task->ptr_matrix = new int[M*M]; 

    #pragma omp parallel num_threads(_thread_task)
    {
      #pragma omp for
      for (int i = 0; i < M; i++) {    
        for (int j = 0; j < M; j++) {   
            (task->ptr_matrix)[i*M+j] = 0;    
            for (int k = 0; k < N; k++) {    
                (task->ptr_matrix)[i*M+j] += (da[i*N+k] * db[k*M+j]);    
            }    
        }    
      }    
    }

    // i am a node with no child
    // delete the pointer directlyd
    if (task->num_child == 0) {
      delete task->ptr_matrix;
    }
  }

  else {
    task->sycl_queue = _sycl_gpu_queues[id];
  
    // GPU version matrix multiplication
    int* da = sycl::malloc_shared<int>(M*N, task->sycl_queue);
    int* db = sycl::malloc_shared<int>(N*M, task->sycl_queue);
    task->ptr_matrix = sycl::malloc_shared<int>(M*M, task->sycl_queue);
    int* dc = task->ptr_matrix;
          
    // initialize matrix a and matrix b
    (task->sycl_queue).parallel_for(
      sycl::range<1>(M*N),
      [=](sycl::id<1> i) {
        da[i] = parent_sum - id;
        db[i] = parent_sum + id;
      }
    ).wait();

    auto _M = (M % 16 == 0) ? M : (M + 16 - M % 16);

    // matrix multiplication c = a * b
    task->sycl_queue.parallel_for(
      sycl::nd_range<2>{sycl::range<2>(_M, _M), sycl::range<2>(16, 16)},
      [=](sycl::nd_item<2> item) {
      
        int row = item.get_global_id(0);
        int col = item.get_global_id(1);
        
        if(row < M && col < M) {
          int sum = 0;
          
          for(int n = 0; n < N; n++) {
              sum += da[row * N + n] * db[n * M + col];
          }
          //task->ptr_matrix[row * M + col] = sum;
          dc[row * M + col] = sum;
        }
      }
    ).wait();

    sycl::free(da, task->sycl_queue);
    sycl::free(db, task->sycl_queue);
 
    // i am the node with no child
    // directly free the pointer 
    if (task->num_child == 0) {
      sycl::free(task->ptr_matrix, task->sycl_queue);
    }
  }
 
  
  /*  
  // declare three USM pointers to three matrixes
  // da points to matrix a, db to matrix b, dc to matrix c
  // GPU version
  int* da = sycl::malloc_shared<int>(M*N, task->sycl_queue);
  int* db = sycl::malloc_shared<int>(N*M, task->sycl_queue);
  task->ptr_matrix = sycl::malloc_shared<int>(M*M, task->sycl_queue);
  int* dc = task->ptr_matrix;
	
  // initialize matrix a and matrix b
  // TODO: combine the two kernels
  //q.parallel_for(
  (task->sycl_queue).parallel_for(
    sycl::range<1>(M*N),
    [=](sycl::id<1> i) {
      da[i] = parent_sum - id;
      db[i] = parent_sum + id;
    }
  ).wait();

  auto _M = (M % 16 == 0) ? M : (M + 16 - M % 16);

  // matrix multiplication c = a * b
  //q.parallel_for(
  task->sycl_queue.parallel_for(
    sycl::nd_range<2>{sycl::range<2>(_M, _M), sycl::range<2>(16, 16)},
    [=](sycl::nd_item<2> item) {
    
      int row = item.get_global_id(0);
      int col = item.get_global_id(1);
      
      if(row < M && col < M) {
        int sum = 0;
        
        for(int n = 0; n < N; n++) {
            sum += da[row * N + n] * db[n * M + col];
        }
        //task->ptr_matrix[row * M + col] = sum;
        dc[row * M + col] = sum;
      }
    }
  ).wait();

  sycl::free(da, task->sycl_queue);
  sycl::free(db, task->sycl_queue);
  */



  // decrement the dependencies
  for (auto& tid : _tgs->_graph[task->ID]) {
    if(_tgs->_tasks[tid]->join_counter.fetch_sub(1)==1){
      enqueue(_tgs->_tasks[tid].get());
    }  
  }

  // finish all tasks
  if (_processed.fetch_add(1) + 1 == _tgs->_V) {
    _stop = true;
    for(auto& cv : _cvs) {
      cv.notify_one();
    }
  }
}




} // end of namespace tgs
