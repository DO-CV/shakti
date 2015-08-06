#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>

#include <DO/Shakti/MultiArray.hpp>

#include "ImageProcessing.hpp"

#include "image_processing.hpp"


using namespace std;


namespace DO { namespace Shakti { namespace cuda {

  template <typename T>
  class Array
  {
    using self_type = Array;

  public:
    Array() = default;

    inline
    Array(const Vector2i& sizes)
    {
      cudaChannelFormatDesc channel_descriptor = cudaCreateChannelDesc<T>();
      CHECK_CUDA_RUNTIME_ERROR(cudaMallocArray(
        &_array, &channel_descriptor, sizes(0), sizes(1)));
    }

    inline
    Array(const T *host_data, const Vector2i& sizes)
      : self_type{ sizes }
    {
      CHECK_CUDA_RUNTIME_ERROR(cudaMemcpyToArray(
        _array, 0, 0, host_data, sizes(0)*sizes(1)*sizeof(T),
        cudaMemcpyHostToDevice));
    }

    inline
    ~Array()
    {
      CHECK_CUDA_RUNTIME_ERROR(cudaFreeArray(_array));
    }

    inline
    operator cudaArray *() const
    {
      return _array;
    }

  protected:
    cudaArray *_array = nullptr;
  };

} /* namespace cuda */
} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti { namespace {

  static Sara::Timer timer;

  inline void tic()
  {
#ifdef SHAKTI_PROFILING
    timer.restart();
#endif
  }

  inline void toc(const char *what)
  {
#ifdef SHAKTI_PROFILING
    auto time = timer.elapsedMs();
    cout << "[" << what << "] Elapsed time = " << time << " ms" << endl;
#endif
  }

} /* namespace profiling */
} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  void gradient(float *out_ptr, const float *in_ptr, const int *sizes)
  {
    const dim3 block_size{ 16, 16 };
    const dim3 grid_size{
      (sizes[0] + block_size.x - 1) / block_size.x,
      (sizes[1] + block_size.y - 1) / block_size.y
    };

    tic();
    cuda::Array<float> in{ in_ptr, { sizes[0], sizes[1] } };
    MultiArray<float, 2> out{ { sizes[0], sizes[1] } };
    toc("Host to device transfer");

    tic();
    CHECK_CUDA_RUNTIME_ERROR(cudaBindTextureToArray(in_texture, in));
    gradient<float><<<grid_size, block_size>>>(out.data(), static_cast<int>(out.size()));
    CHECK_CUDA_RUNTIME_ERROR(cudaUnbindTexture(in_texture));
    toc("CUDA gradient kernel");

    tic();
    out.copy_to_host(out_ptr);
    toc("Device to host transfer");
  }

} /* namespace Shakti */
} /* namespace DO */