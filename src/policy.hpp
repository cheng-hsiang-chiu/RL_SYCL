#pragma once

#include <utility>
#include "tgs.hpp"


namespace tgs {

enum Accelerator {
  GPU = 0,
  CPU
};



class RL_Policy {

public:
  RL_Policy() = default;

  template<typename T>
  std::pair<size_t, Accelerator> policy(T&&);
};


template<typename T>
inline std::pair<size_t, Accelerator>
RL_Policy::policy(T&& task) {
  std::cout << task->ID << ' '
            << task->M  << ' '
            << task->N  << ' '
            << task->dependency << '\n';
  
  size_t thread_id = 0;
  return std::make_pair(thread_id, Accelerator::GPU); 
}








}
