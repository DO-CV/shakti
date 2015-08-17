#pragma once

#include <driver_types.h>


namespace DO { namespace Shakti {

  class Timer
  {
  public:
    Timer();

    ~Timer();

    void restart();

    float elapsed_ms();

  private:
    cudaEvent_t _start;
    cudaEvent_t _stop;
  };

  void tic();

  void toc(const char *what);

} /* namespace Shakti */
} /* namespace DO */
