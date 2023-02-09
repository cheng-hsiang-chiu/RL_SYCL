#include <iostream>
#include <fstream>
#include "tgs.hpp"

int main() {

  //tgs::TGS scheduler(std::thread::hardware_concurrency());
  
  tgs::TGS scheduler(4);

  //scheduler.dump(std::cout);

  scheduler.schedule();

  // dump scheduling results
  scheduler.dump_scheduling(std::cout);
   
  // dump scheduling results to a file, scheduling_result.txt in build directory
  //std::ofstream myfile;
  //myfile.open ("./scheduling_result.txt");
  //scheduler.dump_scheduling(myfile);
  //myfile.close();
  
  return 0;
}
