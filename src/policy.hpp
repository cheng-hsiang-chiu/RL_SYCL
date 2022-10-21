#pragma once

#include <utility>
#include <random>
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


// TODO:
template<typename T>
inline std::pair<size_t, Accelerator>
RL_Policy::policy(T&& task) {
  //std::cout << task->ID << ' '
  //          << task->M  << ' '
  //          << task->N  << ' '
  //          << task->join_counter.load() << '\n';
  
  static std::minstd_rand engine{std::random_device{}()};
  static std::uniform_int_distribution<size_t> distribution;
  size_t thread_id = distribution(engine) % 4;

  Accelerator accelerator;
  switch(distribution(engine)%2) {
    case 0:
      accelerator = Accelerator::GPU;
    break;
    case 1:
      accelerator = Accelerator::CPU;
    break;
  }

  return std::make_pair(thread_id, accelerator); 
}








}
