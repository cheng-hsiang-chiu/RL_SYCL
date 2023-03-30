#pragma once

#include <utility>
#include <random>
#include <fstream>
#include <string>
#include "tgs.hpp"


namespace tgs {

enum Accelerator {
  GPU = 0,
  CPU
};



class RL_Policy {

public:
  //RL_Policy() = default;

  RL_Policy(const size_t n): _num_threads{n}, read_vector(3) {}

  template<typename T>
  std::pair<size_t, Accelerator> policy_read(T&&);

  template<typename T, typename C>
  void state_write(T*, C*);

  ~RL_Policy();

private:
  size_t _num_threads;

  std::string _file_path{"./action_state.txt"};

  std::vector<std::string> read_vector;
};


// TODO:
template<typename T>
inline std::pair<size_t, Accelerator>
RL_Policy::policy_read(T&& task) {
  //std::cout << task->ID << ' '
  //          << task->M  << ' '
  //          << task->N  << ' '
  //          << task->join_counter.load() << '\n';
  
  //static std::minstd_rand engine{std::random_device{}()};
  //static std::uniform_int_distribution<size_t> distribution;
  //size_t thread_id = distribution(engine) % _num_threads;

  std::string policystring;
  
  size_t index = 0;

  read_vector[0] = "";
  read_vector[1] = "";
  read_vector[2] = "";

  while(1) {
    std::ifstream handler(_file_path);
    while (std::getline(handler, policystring)) {
      read_vector[index++] = policystring;  
      if (index == 3) {
        break;
      }
      //std::stringstream policystream(policystring);
      //while (std::getline(policystream, policystring, ' ')) {
      //  policystring >> thread_id >> accelerator;  
      //}
    }
    if (read_vector[2].compare("ACTION_READY") != 0) {
      //std::cout << read_vector[2] << '\n';
      index = 0;
      //handler.close();
      continue;
    }

    handler.close();
    break;
  }
  
  std::cout << "read_vector[0] = " << read_vector[0] << '\n';
  std::cout << "read_vector[1] = " << read_vector[1] << '\n';
  std::cout << "read_vector[2] = " << read_vector[2] << '\n';
  
  size_t thread_id = std::stoi(read_vector[0]);
  Accelerator accelerator;
  switch(std::stoi(read_vector[1])) {
    case 0:
      accelerator = Accelerator::GPU;
    break;
    case 1:
      accelerator = Accelerator::CPU;
    break;
  }
  
  return std::make_pair(thread_id, accelerator); 
}

template<typename T, typename C>
inline void
RL_Policy::state_write(T* tp, C* tgs) {

  std::ofstream handler(_file_path);
  //handler << "0\n";
  //handler << "1\n";
  //handler << "2\n";
  //handler << "3\n";
  //handler << "4\n";
  //handler << "5\n";
  //handler << "6\n";
  //handler << "STATE_READY\n";

  for (size_t i = 0; i < tp->_state_action_tuples.size(); ++i) {
    for (size_t j = 0; j < std::get<0>(tp->_state_action_tuples[i]).size(); ++j) {
      handler << std::get<0>(tp->_state_action_tuples[i])[j].sum_task_loads << " ";
      size_t sum_loading_pid = 0;
      for (auto& pid : (tgs->_tasks)[std::get<2>(tp->_state_action_tuples[i]).tid]->parent_id) {
        if (tgs->_tasks[pid]->worker_id == j) {
          sum_loading_pid += (tgs->_tasks[pid]->M*tgs->_tasks[pid]->M*tgs->_tasks[pid]->N);   
        }
      } 
      handler << sum_loading_pid << ' ';   
    }

    handler << std::get<2>(tp->_state_action_tuples[i]).tid << " "
            << std::get<2>(tp->_state_action_tuples[i]).wid << "\n";
  }
  
  handler << "STATE_READY\n";

  handler.close();
}


inline RL_Policy::~RL_Policy() {
  std::ofstream handler(_file_path);
  handler << "DONE";
  handler.close();
}




}
