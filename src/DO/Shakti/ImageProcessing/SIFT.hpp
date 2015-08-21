#ifndef DO_SHAKTI_IMAGEPROCESSING_SIFT_HPP
#define DO_SHAKTI_IMAGEPROCESSING_SIFT_HPP

#include <DO/Shakti/MultiArray.hpp>


namespace DO { namespace Shakti {

  class DenseSiftComputer
  {
  public:
    DenseSiftComputer();

    MultiArray<Vector<float, 128>, 2>
    operator()(const Cuda::Array<Vector2f>& gradients) const;

    void operator()(float *out, const float *in, const int *sizes) const;

  private:
    float _bin_scale_unit_length = 3.f;
    //! \brief Maximum value for a descriptor bin value to remain robust w.r.t.
    //! illumination changes.
    float _max_bin_value = 0.2f;
    float _sigma = 1.6f;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_SIFT_HPP */