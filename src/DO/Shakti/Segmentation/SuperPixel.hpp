#ifndef DO_SHAKTI_SEGMENTATION_SUPERPIXEL_HPP
#define DO_SHAKTI_SEGMENTATION_SUPERPIXEL_HPP

#include <DO/Shakti/MultiArray/Matrix.hpp>
#include <DO/Shakti/MultiArray/MultiArray.hpp>


namespace DO { namespace Shakti {

  struct Cluster
  {
    Vector3f color;
    Vector2f center;
    int num_points;
  };

  class SegmentationSLIC
  {
  public:
    //! \brief Constructor.
    SegmentationSLIC() = default;

    //! @{
    //! \brief Getters.
    Vector2i get_image_sizes() const;

    int get_image_padded_width() const;
    //! @}

    //! @{
    //! \brief Setters.
    void set_image_sizes(const Vector2i& sizes) const;

    void set_image_padded_width(int padded_width) const;

    void set_image_sizes(const MultiArray<Vector3f, 2>& device_image) const
    {
      set_image_padded_width(device_image.padded_width());
      set_image_sizes(device_image.sizes());
    }
    //! @}

  public:
    //! @{
    //! \brief Run the algorithm.
    MultiArray<int, 2> operator()(const MultiArray<Vector3f, 2>& image) const;

    void operator()(int *out_labels, const Vector3f *rgb_image, const int *sizes) const;
    //! @}

  public:
    MultiArray<Cluster, 2> init_clusters(const MultiArray<Vector3f, 2>& image) const;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_SEGMENTATION_SUPERPIXEL_HPP */