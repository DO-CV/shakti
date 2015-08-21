#ifndef DO_SHAKTI_IMAGEPROCESSING_CUDA_DIFFERENTIAL_HPP
#define DO_SHAKTI_IMAGEPROCESSING_CUDA_DIFFERENTIAL_HPP


namespace DO { namespace Shakti {

  __global__
  void compute_gradients(Vector2f *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    dst[i] = {
      tex2D(in_float_texture, p(0) + 1, p(1)) - tex2D(in_float_texture, p(0) - 1, p(1)),
      tex2D(in_float_texture, p(0), p(1) + 1) - tex2D(in_float_texture, p(0), p(1) - 1)
    };
  }

  __global__
  void compute_gradient_polar_coordinates(Vector2f *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    const auto f_x = tex2D(in_float_texture, p(0) + 1, p(1)) - tex2D(in_float_texture, p(0) - 1, p(1));
    const auto f_y = tex2D(in_float_texture, p(0), p(1) + 1) - tex2D(in_float_texture, p(0), p(1) - 1);

    dst[i] = {
      sqrt(f_x*f_x + f_y*f_y),
      atan2(f_y, f_x)
    };
  }

  __global__
  void compute_squared_norms(float *out, const Vector<float, 2> *in)
  {
    const auto i = offset<2>();
    const auto f_i = in[i];

    out[i] = f_i.squared_norm();
  }

  __global__
  void compute_gradient_squared_norms(float *dst)
  {
    const auto i = offset<2>();
    const auto p = coords<2>();

    auto u_x = tex2D(in_float_texture, p(0) + 1, p(1)) - tex2D(in_float_texture, p(0) - 1, p(1));
    auto u_y = tex2D(in_float_texture, p(0), p(1) + 1) - tex2D(in_float_texture, p(0), p(1) - 1);
    dst[i] = u_x*u_x + u_y*u_y;
  }

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_CUDA_DIFFERENTIAL_HPP */