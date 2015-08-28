#include <DO/Shakti/MultiArray.hpp>


#define MAX_BLOCK_SIZE 256


namespace DO { namespace Shakti {

  struct Cluster
  {
    Vector3f color;
    Vector2f center;
    int num_points;
    Vector2f bbox_p1;
    Vector2f bbox_p2;
  };

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
    const Vector2f c_b = tl_b + cluster_sizes / 2;
    // The image offset of the block center is:
    const o_c_b = c_b.x() + c_b.y() * image_padded_width;

    out_clusters[i_b].num_points = 0;
    out_clusters[i_b].center = c_b;
    out_clusters[i_b].color = in_image[o_c_b];

  }

  __device__
  float squared_distance(const Vector2f& x1, const Vector3f& I1, const Vector2f& x2, const Vector3f& I2)
  {
    // TODO.
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
    float closest_distance{ std::numeric_limits<float>::max() };
#pragma unroll
    for (int x_b = x1_b; x_b < x2_b; ++x_b)
    {
#pragma unroll
      for (int y_b = x1_b; y_b < y2_b; ++y_b)
      {
        auto d = distance(p_c[x_b][y_b], I_c[x_b][y_b], p_i, I_i);
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
    out_clusters
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  MultiArray<int, 2> slic_segment(const MultiArray<Vector3f, 2>& image)
  {
    // Create the image of labels.
    MultiArray<int, 2> labels{ image.sizes() };

    // Create the grid for CUDA.
    const auto image_block_sizes = default_block_size_2d();
    const auto image_grid_sizes = default_grid_size_2d(labels);

    // Compute the number of clusters.
    const auto& sizes = image.sizes();
    const auto padded_width = image.padded_width();
    const auto& num_clusters_2d = Vector2i{
      (sizes.x() + image_block_sizes.x - 1) / image_block_sizes.x, // number of clusters per columns
      (sizes.y() + image_block_sizes.y - 1) / image_block_sizes.y // number of clusters per rows
    };
    const auto num_clusters = num_clusters_2d.x() * num_clusters_2d.y();

    const auto cluster_block_sizes = 16;
    const auto cluster_grid_sizes = (num_clusters + cluster_block_sizes - 1) / cluster_block_sizes;

    MultiArray<Cluster, 1> clusters{ num_clusters_2d };

    // Init clusters.
    cudaMemcpyToSymbol(&image_sizes, image.sizes().data(), sizeof(Vector2i));
    cudaMemcpyToSymbol(&image_padded_width, &padded_width, sizeof(int));
    cudaMemcpyToSymbol(&num_clusters, num_clusters_2d.data(), sizeof(Vector2i));
    init_clusters<<<cluster_grid_sizes, cluster_block_sizes>>>(clusters.data(), image.data());

    // Iterate.
    for (int i = 0; i < 5; ++i)
    {
      assign_means<<<image_grid_sizes, image_block_sizes>>>(labels.data(), image.data(), clusters.data());
      update_means<<<cluster_grid_sizes, cluster_block_sizes>>>(clusters.data(), labels.data(), image.data());
    }

    return labels;
  }

} /* namespace Shakti */
} /* namespace DO */