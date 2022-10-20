#pragma once

#include <utility>
#include <random>
#include "tgs.hpp"


namespace tgs {

enum Accelerator {
  GPU = 0,
  CPU
};

static std::minstd_rand engine{std::random_device{}()};
static std::uniform_int_distribution<size_t> distribution;
static size_t random_value() {
  return distribution(engine);
}

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
            << task->join_counter.load() << '\n';
  
  size_t thread_id = random_value()%6;
  return std::make_pair(thread_id, Accelerator::GPU); 
}








}
