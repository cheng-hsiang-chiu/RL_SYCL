#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"

namespace tgs {


size_t getGPUMemUsage(size_t GPU_ID) {
  cudaSetDevice(GPU_ID);
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  
  return (total-free);
}


}
  
