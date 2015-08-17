#pragma once

#include <DO/Shakti/MultiArray/Matrix.hpp>


namespace DO { namespace Shakti { namespace Cuda {

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
    Array(const T *data, const Vector2i& sizes,
          cudaMemcpyKind kind = cudaMemcpyHostToDevice)
      : self_type{ sizes }
    {
      copy_from(data, sizes, kind);
    }

    inline
    ~Array()
    {
      CHECK_CUDA_RUNTIME_ERROR(cudaFreeArray(_array));
    }

    inline
    void copy_from(const T *data, const Vector2i& sizes, cudaMemcpyKind kind)
    {
      CHECK_CUDA_RUNTIME_ERROR(cudaMemcpyToArray(
        _array, 0, 0, data, sizes(0)*sizes(1)*sizeof(T), kind));
    }

    operator cudaArray *() const
    {
      return _array;
    }

  protected:
    cudaArray *_array = nullptr;
  };

} /* namespace Cuda */
} /* namespace Shakti */
} /* namespace DO */
