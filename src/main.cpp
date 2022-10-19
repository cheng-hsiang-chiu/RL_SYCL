#include <iostream>
#include "tgs.hpp"

int main() {


  tgs::TGS scheduler(std::thread::hardware_concurrency());

  scheduler.dump(std::cout);

  scheduler.schedule();

  return 0;
}
