#include <cuda_runtime_api.h>

#include <iostream>

#include <DO/Shakti/Utilities/ErrorCheck.hpp>
#include <DO/Shakti/Utilities/Timer.hpp>


namespace DO { namespace Shakti {

  Timer::Timer()
  {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  Timer::~Timer()
  {
    CHECK_CUDA_RUNTIME_ERROR(cudaEventDestroy(_start));
    CHECK_CUDA_RUNTIME_ERROR(cudaEventDestroy(_stop));
  }

  void Timer::restart()
  {
    cudaEventRecord(_start);
  }

  float Timer::elapsed_ms()
  {
    float ms;
    CHECK_CUDA_RUNTIME_ERROR(cudaEventRecord(_stop));
    CHECK_CUDA_RUNTIME_ERROR(cudaEventSynchronize(_stop));
    CHECK_CUDA_RUNTIME_ERROR(cudaEventElapsedTime(&ms, _start, _stop));
    return ms;
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  static Timer timer;

  void tic()
  {
    timer.restart();
  }

  void toc(const char *what)
  {
    auto time = timer.elapsed_ms();
    std::cout << "[" << what << "] Elapsed time = " << time << " ms" << std::endl;
  }

} /* namespace Shakti */
} /* namespace DO */
