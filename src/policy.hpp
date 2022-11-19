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
//  RL_Policy() = default;

  RL_Policy(const size_t n): _num_threads{n} {}

  template<typename T>
  std::pair<size_t, Accelerator> policy(T&&);

private:
  size_t _num_threads;
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
  size_t thread_id = distribution(engine) % _num_threads;
  return std::make_pair(thread_id, Accelerator::GPU); 
}








}
