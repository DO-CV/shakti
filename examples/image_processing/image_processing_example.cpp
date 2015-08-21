#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/MultiArray/Cuda/Array.hpp>
#include <DO/Shakti/Utilities/DeviceInfo.hpp>
#include <DO/Shakti/Utilities/Timer.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


using namespace std;
using namespace DO;
using namespace sara;


template <int N, int O>
void draw_grid(float x, float y, float sigma, float theta, int pen_width = 1)
{
  const float lambda = 3.f;
  const float l = lambda*sigma;
  Vector2f grid[N + 1][N + 1];
  Matrix2f T;
  theta = 0;
  T << cos(theta), -sin(theta),
    sin(theta), cos(theta);
  T *= l;
  for (int v = 0; v < N + 1; ++v)
    for (int u = 0; u < N + 1; ++u)
      grid[u][v] = (Vector2f{ x, y } +T*Vector2f{ u - N / 2.f, v - N / 2.f });
  for (int i = 0; i < N + 1; ++i)
    draw_line(grid[0][i], grid[N][i], Green8, pen_width);
  for (int i = 0; i < N + 1; ++i)
    draw_line(grid[i][0], grid[i][N], Green8, pen_width);

  Vector2f a(x, y);
  Vector2f b;
  b = a + N / 2.f*T*Vector2f(1, 0);
  draw_line(a, b, Red8, pen_width + 2);
}


GRAPHICS_MAIN()
{
  try
  {
    auto devices = shakti::get_devices();
    devices.front().make_current_device();

    VideoStream video_stream{
      "C:/Users/David/Desktop/GitHub/sara/examples/VideoIO/orion_1.mpg"
    };
    auto video_frame_index = int{ 0 };
    auto video_frame = Image<Rgb8>{};

    auto in_frame = Image<float>{};
    auto out_frame = Image<float>{};
    auto apply_gaussian_filter = shakti::GaussianFilter{ 1.6f };

    auto sifts = Image<Vector128f>{};
    auto sift_computer = shakti::DenseSiftComputer{};

    while (video_stream.read(video_frame))
    {
      cout << "[Read frame] " << video_frame_index << "" << endl;
      if (!active_window())
        create_window(video_frame.sizes());

      in_frame = video_frame.convert<float>();
      out_frame.resize(in_frame.sizes());

      shakti::tic();
      apply_gaussian_filter(out_frame.data(), in_frame.data(), in_frame.sizes().data());
      shakti::toc("Gaussian filter");

      //shakti::tic();
      //shakti::compute_gradient_squared_norms(out_frame.data(), out_frame.data(), in_frame.sizes().data());
      //shakti::toc("Gradient norms");

      sifts.resize(in_frame.sizes());
      sift_computer(
        reinterpret_cast<float *>(sifts.data()), in_frame.data(),
        in_frame.sizes().data());

      display(out_frame);

      //CHECK(sifts(0, 0));

      //get_key();


      ++video_frame_index;
      cout << endl;
    }
  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
  }

  return 0;
}