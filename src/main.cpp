#include <iostream>
#include <fstream>
#include "tgs.hpp"

int main() {

  //tgs::TGS scheduler(std::thread::hardware_concurrency());
  
  // 4 is the number of worker threads
  // 16 is the number of thread for a CPU task
  tgs::TGS scheduler(4, 16);

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
