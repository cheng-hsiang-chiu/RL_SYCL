#include <iostream>
#include "tgs.hpp"

int main() {


  tgs::TGS scheduler(std::thread::hardware_concurrency());

  scheduler.dump(std::cout);


  return 0;
}
