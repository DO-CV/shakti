#pragma once

#include <DO/Shakti/MultiArray.hpp>


namespace DO { namespace Shakti {

  void apply_x_derivative(float *out, const float *in, const int *sizes);

  void apply_y_derivative(float *out, const float *in, const int *sizes);

  class GaussianFilter
  {
  public:
    GaussianFilter(float sigma, int truncation_factor = 4.f)
      : _truncation_factor{ truncation_factor }
    {
      set_sigma(sigma);
    }

    void set_sigma(float sigma);

    void operator()(float *out, const float *in, const int *sizes) const;

  private:
    int _truncation_factor;
    std::vector<float> _kernel;
  };

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  template <typename T, int N>
  class ImagePyramid
  {
  public:
    using scalar_type = T;

    ImagePyramid(const Vector2i& image_sizes);

  private:
    scalar_type _scale_initial;
    scalar_type _scale_geometric_factor;
    T *_device_data;
  };

} /* namespace Shakti */
} /* namespace DO */
