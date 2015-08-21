#ifndef DO_SHAKTI_IMAGEPROCESSING_LINEARFILTERING_HPP
#define DO_SHAKTI_IMAGEPROCESSING_LINEARFILTERING_HPP

#include <vector>

#include <DO/Shakti/MultiArray/Cuda/Array.hpp>
#include <DO/Shakti/MultiArray/MultiArray.hpp>


namespace DO { namespace Shakti {

  void apply_row_based_convolution(
    float *out, const float *in, const float *kernel,
    int kernel_size, const int *sizes);

  void apply_column_based_convolution(
    float *out, const float *in, const float *kernel,
    int kernel_size, const int *sizes);

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

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

    MultiArray<float, 2> operator()(Cuda::Array<float>& in) const;

  private:
    int _truncation_factor;
    std::vector<float> _kernel;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_LINEARFILTERING_HPP */