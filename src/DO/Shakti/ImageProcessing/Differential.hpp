#ifndef DO_SHAKTI_IMAGEPROCESSING_DIFFERENTIAL_HPP
#define DO_SHAKTI_IMAGEPROCESSING_DIFFERENTIAL_HPP

#include <DO/Shakti/MultiArray/Cuda/Array.hpp>
#include <DO/Shakti/MultiArray/MultiArray.hpp>


namespace DO { namespace Shakti {

  MultiArray<Vector2f, 2> gradient(const Cuda::Array<float>& in);

  MultiArray<Vector2f, 2> gradient_polar_coords(const Cuda::Array<float>& in);

  MultiArray<float, 2> gradient_squared_norm(const Cuda::Array<float>& in);

  MultiArray<float, 2> squared_norm(const MultiArray<Vector2f, 2>& in);

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  void compute_x_derivative(float *out, const float *in, const int *sizes);

  void compute_y_derivative(float *out, const float *in, const int *sizes);

  void compute_gradient_squared_norms(float *out, const float *in, const int *sizes);

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_DIFFERENTIAL_HPP */