#pragma once

#include <DO/Shakti/MultiArray/Matrix.hpp>


namespace DO { namespace Shakti { namespace Cuda {

  template <typename T>
  struct ChannelFormatDescriptor
  {
    static inline cudaChannelFormatDesc type()
    {
      return cudaCreateChannelDesc<T>();
    }
  };

  template <>
  struct ChannelFormatDescriptor<Vector2f>
  {
    static inline cudaChannelFormatDesc type()
    {
      cudaChannelFormatDesc format = {
        32, 32, 0, 0, cudaChannelFormatKindFloat
      };
      return format;
    }
  };


  template <typename T>
  class Array
  {
    using self_type = Array;

  public:
    Array() = default;

    inline
    Array(const Vector2i& sizes)
      : _sizes{ sizes }
    {
      auto channel_descriptor = ChannelFormatDescriptor<T>::type();
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

    inline
    operator cudaArray *() const
    {
      return _array;
    }

    inline
    const Vector2i& sizes() const
    {
      return _sizes;
    }

  protected:
    cudaArray *_array = nullptr;
    Vector2i _sizes = Vector2i::Zero();
  };

} /* namespace Cuda */
} /* namespace Shakti */
} /* namespace DO */