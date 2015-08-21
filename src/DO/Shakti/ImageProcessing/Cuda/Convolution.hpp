#ifndef DO_SHAKTI_IMAGEPROCESSING_CUDA_CONVOLUTION_HPP
#define DO_SHAKTI_IMAGEPROCESSING_CUDA_CONVOLUTION_HPP

#include <DO/Shakti/ImageProcessing/Cuda/Globals.hpp>

#include <DO/Shakti/MultiArray/Matrix.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>


namespace DO { namespace Shakti {

  template <typename T>
  __global__
  void apply_column_based_convolution(T *dst)
  {
    const int i{ offset<2>() };
    const Vector2i p{ coords<2>() };

    auto convolved_value = T{ 0 };
    auto kernel_radius = convolution_kernel_size / 2;
#pragma unroll
    for (int i = 0; i < convolution_kernel_size; ++i)
      convolved_value += tex2D(in_float_texture, p(0) - kernel_radius + i, p(1)) * convolution_kernel[i];
    dst[i] = convolved_value;
  }

  template <typename T>
  __global__
  void apply_row_based_convolution(T *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    auto convolved_value = T{ 0 };
    auto kernel_radius = convolution_kernel_size / 2;
#pragma unroll
    for (int i = 0; i< convolution_kernel_size; ++i)
      convolved_value += tex2D(in_float_texture, p(0), p(1) - kernel_radius + i) * convolution_kernel[i];
    dst[i] = convolved_value;
  }

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_CUDA_CONVOLUTION_HPP */