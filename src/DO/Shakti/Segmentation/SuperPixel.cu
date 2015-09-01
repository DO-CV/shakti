#include <math_constants.h>

#include <DO/Shakti/MultiArray/Grid.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>

#include <DO/Shakti/Segmentation/SuperPixel.hpp>


#define MAX_BLOCK_SIZE 256


namespace DO { namespace Shakti {

  __constant__ Vector2i image_sizes;
  __constant__ int image_padded_width;

  __constant__ Vector2i num_clusters;
  __constant__ Vector2i cluster_sizes;

  __constant__ float distance_weight;


  // At the beginning a cluster corresponds to a block in a grid.
  __global__
  void init_clusters(Cluster *out_clusters, const Vector3f *in_image)
  {
    // The 1D index of the block is:
    const auto i_b = blockIdx.x * blockDim.x + threadIdx.x;

    // The 2D indices of the block is:
    const auto x_b = i_b % num_clusters.x();
    const auto y_b = i_b / num_clusters.y();

    // The image coordinates of the top-left corner of the block is:
    const auto tl_b = Vector2f{
      x_b*cluster_sizes.x(),
      y_b*cluster_sizes.y()
    };

    // The 2D image coordinates of the block center is:
    const Vector2f c_b = tl_b + Vector2f{ cluster_sizes.x() / 2, cluster_sizes.y() / 2 };
    // The image offset of the block center is:
    const int o_c_b = c_b.x() + c_b.y() * image_padded_width;

    out_clusters[i_b].num_points = 0;
    out_clusters[i_b].center = c_b;
    out_clusters[i_b].color = in_image[o_c_b];

  }

  __device__
  float squared_distance(const Vector2f& x1, const Vector3f& I1,
                         const Vector2f& x2, const Vector3f& I2)
  {
    // TODO.
    return 0.f;
  }

  __global__
  void assign_means(int *out_labels, const Vector3f *in_image, const Cluster *in_clusters)
  {
    // For each thread in the block, populate the list of nearest cluster centers.
    __shared__ int x1, x2, y1, y2;
    __shared__ int x1_b, x2_b, y1_b, y2_b;
    __shared__ Vector3f I_c[3][3]; // I(c) = color value of a cluster center $c$.
    __shared__ Vector2f p_c[3][3]; // p(c) = coordinates of cluster center $c$.

    // Pixel indices.
    __shared__ Cluster pixelUpdateList[MAX_BLOCK_SIZE];
    __shared__ Vector2f pixelUpdateIdx[MAX_BLOCK_SIZE];

    // Get the pixel 2D coordinates, offset and color value.
    const auto _p_i = coords<2>();
    // Don't consider padded region of the image.
    if (_p_i.x() >= image_sizes.x() || _p_i.y() >= image_sizes.y())
      return;
    const auto p_i = Vector2f{ _p_i.x(), _p_i.y() };
    const auto i = offset<2>();
    const auto I_i = in_image[i];

    // For each pixel (x, y), find the spatially nearest cluster centers.
    // In each block, the 3x3 top-left threads will populate the list of the cluster centers.
    // The other threads will wait.
    if (threadIdx.x < 3 || threadIdx.y < 3)
    {
      // Take care of the boundary case.
      x1 = blockIdx.x == 0 ? 1 : 0;
      y1 = blockIdx.y == 0 ? 1 : 0;
      x2 = blockIdx.x >= num_clusters.x() ? 1 : 2;
      y2 = blockIdx.y >= num_clusters.y() ? 1 : 2;

      // Retrieve the corresponding blocks.
      x1_b = blockIdx.x + x1 - 1;
      y1_b = blockIdx.y + y1 - 1;

      x2_b = blockIdx.x + x2 - 1;
      y2_b = blockIdx.y + y2 - 1;

#pragma unroll
      for (int x_b = x1_b; x_b < x2_b; ++x_b)
      {
#pragma unroll
        for (int y_b = x1_b; y_b < y2_b; ++y_b)
        {
          I_c[x_b][y_b] = in_clusters[x_b + y_b * cluster_sizes.x()].color;
          p_c[x_b][y_b] = in_clusters[x_b + y_b * cluster_sizes.x()].center;
        }
      }
    }

    __syncthreads();


    // Assign the closest centers.
    Vector2i closest_cluster_index{};
    float closest_distance{ CUDART_INF_F };
#pragma unroll
    for (int x_b = x1_b; x_b < x2_b; ++x_b)
    {
#pragma unroll
      for (int y_b = x1_b; y_b < y2_b; ++y_b)
      {
        auto d = squared_distance(p_c[x_b][y_b], I_c[x_b][y_b], p_i, I_i);
        if (d < closest_distance)
        {
          closest_distance = d;
          closest_cluster_index = Vector2i{ x_b, y_b };
        }
      }
    }

    const auto cluster_offset = closest_cluster_index.x() + closest_cluster_index.y() * num_clusters.x();
    out_labels[i] = cluster_offset;
  }

  __global__
  void update_means(Cluster *out_clusters, const int *in_labels, const Vector3f *in_image)
  {
    int blockWidth = image_sizes.x() / blockDim.x;
    int blockHeight = image_sizes.y() / gridDim.x;

    int clusterIdx = blockIdx.x*blockDim.x + threadIdx.x;

    int offsetBlock = threadIdx.x * blockWidth + blockIdx.x * blockHeight * image_sizes.x();

    Vector2f crntXY = out_clusters[clusterIdx].center;
    Vector3f avLab;
    Vector2f avXY;
    int nPoints = 0;

    avLab.x() = 0;
    avLab.y() = 0;
    avLab.z() = 0;

    avXY.x() = 0;
    avXY.y() = 0;

    int yBegin = 0 < (crntXY.y() - blockHeight) ? (crntXY.y() - blockHeight) : 0;
    int yEnd = image_sizes.y() > (crntXY.y() + blockHeight) ? (crntXY.y() + blockHeight) : (image_sizes.y() - 1);
    int xBegin = 0 < (crntXY.x() - blockWidth) ? (crntXY.x() - blockWidth) : 0;
    int xEnd = image_sizes.x() > (crntXY.x() + blockWidth) ? (crntXY.x() + blockWidth) : (image_sizes.x() - 1);

    //update to cluster centers
    for (int i = yBegin; i < yEnd; i++)
    {
      for (int j = xBegin; j < xEnd; j++)
      {
        int offset = j + i * image_sizes.x();

        Vector3f fPixel = in_image[offset];
        int pIdx = in_labels[offset];

        if (pIdx == clusterIdx)
        {
          avLab.x() += fPixel.x();
          avLab.y() += fPixel.y();
          avLab.z() += fPixel.z();

          avXY.x() += j;
          avXY.y() += i;

          nPoints++;
        }
      }
    }

    if (nPoints == 0)
      return;

    avLab.x() /= nPoints;
    avLab.y() /= nPoints;
    avLab.z() /= nPoints;

    avXY.x() /= nPoints;
    avXY.y() /= nPoints;

    out_clusters[clusterIdx].color = avLab;
    out_clusters[clusterIdx].center = avXY;
    out_clusters[clusterIdx].num_points = nPoints;
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  Vector2i SegmentationSLIC::get_image_sizes() const
  {
    auto sizes = Vector2i{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
      &sizes, image_sizes, sizeof(Vector2i)));
    return sizes;
  }

  int SegmentationSLIC::get_image_padded_width() const
  {
    auto padded_width = int{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
      &padded_width, image_padded_width, sizeof(int)));
    return padded_width;
  }

  void SegmentationSLIC::set_image_sizes(const Vector2i& sizes) const
  {
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      image_sizes, &sizes, sizeof(Vector2i)));
  }

  void SegmentationSLIC::set_image_padded_width(int padded_width) const
  {
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      image_padded_width, &padded_width, sizeof(int)));
  }

  MultiArray<int, 2>
  SegmentationSLIC::operator()(const MultiArray<Vector3f, 2>& image) const
  {
    // Create the image of labels.
    MultiArray<int, 2> labels{ image.sizes() };

    // Create the grid for CUDA.
    const auto image_block_sizes = default_block_size_2d();
    const auto image_grid_sizes = default_grid_size_2d(labels);

    // Compute the number of clusters.
    const auto& sizes = image.sizes();
    const auto padded_width = image.padded_width();
    const auto num_clusters_2d = Vector2i(
      (sizes.x() + image_block_sizes.x - 1) / image_block_sizes.x, // number of clusters per columns
      (sizes.y() + image_block_sizes.y - 1) / image_block_sizes.y // number of clusters per rows
    );
    const auto num_clusters = num_clusters_2d.x() * num_clusters_2d.y();

    const auto cluster_block_sizes = 16;
    const auto cluster_grid_sizes = (num_clusters + cluster_block_sizes - 1) / cluster_block_sizes;

    MultiArray<Cluster, 2> clusters{ { num_clusters, 1 } };

    // Init clusters.
    cudaMemcpyToSymbol(&image_sizes, image.sizes().data(), sizeof(Vector2i));
    cudaMemcpyToSymbol(&image_padded_width, &padded_width, sizeof(int));
    cudaMemcpyToSymbol(&num_clusters, num_clusters_2d.data(), sizeof(Vector2i));
    Shakti::init_clusters<<<cluster_grid_sizes, cluster_block_sizes>>>(clusters.data(), image.data());

    // Iterate.
    for (int i = 0; i < 5; ++i)
    {
      assign_means<<<image_grid_sizes, image_block_sizes>>>(labels.data(), image.data(), clusters.data());
      update_means<<<cluster_grid_sizes, cluster_block_sizes>>>(clusters.data(), labels.data(), image.data());
    }

    return labels;
  }

  void
  SegmentationSLIC::operator()(int *labels, const Vector3f *rgb_image, const int *sizes) const
  {
    auto image_array = MultiArray<Vector3f, 2>{ rgb_image, sizes };
    auto labels_array = (*this)(image_array);
    labels_array.copy_to_host(labels);
  }

  MultiArray<Cluster, 2>
  SegmentationSLIC::init_clusters(const MultiArray<Vector3f, 2>& image) const
  {
    MultiArray<Cluster, 2> clusters{ image.sizes() };
    return clusters;
  }

} /* namespace Shakti */
} /* namespace DO */