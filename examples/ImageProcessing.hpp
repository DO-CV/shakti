#pragma once

#include <DO/Shakti/MultiArray/Matrix.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>


namespace DO { namespace Shakti {

  texture<float, 2> in_texture;
  texture<float, 2> out_texture;


  template <typename T>
  __global__
  void gradient(T *dst, int size)
  {
    const int i{ offset<2>() };
    const Vector2i p{ coords<2>() };
    if (i >= size)
      return;

    auto u_x = tex2D(in_texture, p(0)+1, p(1)) - tex2D(in_texture, p(0)-1, p(1));
    auto u_y = tex2D(in_texture, p(0), p(1)+1) - tex2D(in_texture, p(0), p(1)-1);
    dst[i] = sqrt(u_x*u_x + u_y*u_y);
  }

  template <typename T, int N>
  __global__
  void laplacian(const T *src, T *dst, int size)
  {
    int i = DO::Shakti::offset<N>();

    dst[i] = -2 * N*src[i];
  }

} /* namespace Shakti */
} /* namespace DO */