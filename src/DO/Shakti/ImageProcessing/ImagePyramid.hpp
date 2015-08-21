#ifndef DO_SHAKTI_IMAGEPROCESSING_IMAGEPYRAMID_HPP
#define DO_SHAKTI_IMAGEPROCESSING_IMAGEPYRAMID_HPP

#include <DO/Shakti/MultiArray/Matrix.hpp>


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


#endif /* DO_SHAKTI_IMAGEPROCESSING_IMAGEPYRAMID_HPP */