#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <DO/Shakti/Utilities/DeviceInfo.hpp>
#include <DO/Shakti/Utilities/Timer.hpp>

#include "image_processing.hpp"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;

using namespace std;
using namespace DO;
using namespace sara;


GRAPHICS_MAIN()
{
  try
  {
    auto devices = shakti::get_devices();
    devices.front().make_current_device();

    VideoStream video_stream{
      "/home/david/Desktop/GitHub/DO-CV/sara/examples/VideoIO/orion_1.mpg"
    };
    auto video_frame_index = int{ 0 };
    auto video_frame = Image<Rgb8>{};

    auto in_frame = Image<float>{};
    auto out_frame = Image<float>{};
    auto apply_gaussian_filter = shakti::GaussianFilter{ 4.f };

#define DISPLAY
    while (video_stream.read(video_frame))
    {
      cout << "[Read frame] " << video_frame_index << "" << endl;
#ifdef DISPLAY
      if (!active_window())
        create_window(video_frame.sizes());
#endif

      in_frame = video_frame.convert<float>();
      out_frame.resize(in_frame.sizes());

      shakti::tic();
      apply_gaussian_filter(out_frame.data(), in_frame.data(), in_frame.sizes().data());
      shakti::toc("Gaussian filter");

#ifdef DISPLAY
      display(out_frame);
#endif

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
