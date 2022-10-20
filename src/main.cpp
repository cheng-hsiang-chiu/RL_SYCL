#include <iostream>
#include "tgs.hpp"

int main() {

  //tgs::TGS scheduler(std::thread::hardware_concurrency());
  
  tgs::TGS scheduler(4);

  //scheduler.dump(std::cout);

  scheduler.schedule();

  return 0;
}
