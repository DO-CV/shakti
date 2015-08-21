#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/ImageProcessing/Cuda/Convolution.hpp>
#include <DO/Shakti/ImageProcessing/Cuda/Differential.hpp>

#include <DO/Shakti/MultiArray.hpp>


namespace DO { namespace Shakti {

  MultiArray<Vector2f, 2> gradient(const Cuda::Array<float>& in)
  {
    const auto& sizes = in.sizes();
    const dim3 block_size{ 16, 16 };
    const dim3 grid_size{
      (sizes[0] + block_size.x - 1) / block_size.x,
      (sizes[1] + block_size.y - 1) / block_size.y
    };

    MultiArray<Vector2f, 2> dst{ in.sizes() };
    CHECK_CUDA_RUNTIME_ERROR(cudaBindTextureToArray(in_float_texture, in));
    compute_gradients<<<grid_size, block_size>>>(dst.data());
    CHECK_CUDA_RUNTIME_ERROR(cudaUnbindTexture(in_float_texture));

    return dst;
  }

  MultiArray<Vector2f, 2> gradient_polar_coords(const Cuda::Array<float>& in)
  {
    const auto& sizes = in.sizes();
    const dim3 block_size{ 16, 16 };
    const dim3 grid_size{
      (sizes[0] + block_size.x - 1) / block_size.x,
      (sizes[1] + block_size.y - 1) / block_size.y
    };

    MultiArray<Vector2f, 2> out{ in.sizes() };
    CHECK_CUDA_RUNTIME_ERROR(cudaBindTextureToArray(in_float_texture, in));
    compute_gradient_polar_coordinates<<<grid_size, block_size>>>(out.data());
    CHECK_CUDA_RUNTIME_ERROR(cudaUnbindTexture(in_float_texture));

    return out;
  }

  MultiArray<float, 2> gradient_squared_norm(const Cuda::Array<float>& in)
  {
    const auto& sizes = in.sizes();
    const dim3 block_size{ 16, 16 };
    const dim3 grid_size{
      (sizes[0] + block_size.x - 1) / block_size.x,
      (sizes[1] + block_size.y - 1) / block_size.y
    };

    MultiArray<float, 2> out{ in.sizes() };
    CHECK_CUDA_RUNTIME_ERROR(cudaBindTextureToArray(in_float_texture, in));
    compute_gradient_squared_norms<<<grid_size, block_size>>>(out.data());
    CHECK_CUDA_RUNTIME_ERROR(cudaUnbindTexture(in_float_texture));

    return out;
  }

  MultiArray<float, 2> squared_norm(const MultiArray<Vector2f, 2>& in)
  {
    const auto& sizes = in.sizes();
    const dim3 block_size{ 16, 16 };
    const dim3 grid_size{
      (sizes(0) + block_size.x - 1) / block_size.x,
      (sizes(1) + block_size.y - 1) / block_size.y
    };

    MultiArray<float, 2> out{ in.sizes() };
    compute_squared_norms<<<grid_size, block_size>>>(out.data(), in.data());
    return out;
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  void compute_x_derivative(float *out, const float *in, const int *sizes)
  {
    const float kernel[] = { -1.f, 0.f, 1.f };
    const int kernel_size{ 3 };
    apply_column_based_convolution(out, in, kernel, kernel_size, sizes);
  }

  void compute_y_derivative(float *out, const float *in, const int *sizes)
  {
    const float kernel[] = { -1.f, 0.f, 1.f };
    const int kernel_size{ 3 };
    apply_row_based_convolution(out, in, kernel, kernel_size, sizes);
  }

  void compute_gradient_squared_norms(float *out, const float *in, const int *sizes)
  {
    Cuda::Array<float> in_cuda_array{ in, { sizes[0], sizes[1] } };

#ifdef TWO_KERNEL
    MultiArray<Vector2f, 2> gradients{
      gradient(in_cuda_array)
    };
    MultiArray<float, 2> gradient_squared_norms{
      squared_norm(gradients)
    };
#else
    const dim3 block_size{ 16, 16 };
    const dim3 grid_size{
      (sizes[0] + block_size.x - 1) / block_size.x,
      (sizes[1] + block_size.y - 1) / block_size.y
    };

    MultiArray<float, 2> gradient_squared_norms{ gradient_squared_norm(in_cuda_array) };
#endif

    gradient_squared_norms.copy_to_host(out);
  }

} /* namespace Shakti */
} /* namespace DO */
