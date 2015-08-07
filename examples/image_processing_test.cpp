#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <DO/Shakti/Utilities/DeviceInfo.hpp>

#include "image_processing.hpp"

namespace sara = DO::Sara;
namespace shakti = DO::Shakti;

using namespace std;
using namespace DO;
using namespace sara;


namespace {
  static Timer timer;

  void tic()
  {
    timer.restart();
  }

  void toc(const char *what)
  {
    auto time = timer.elapsedMs();
    cout << "[" << what << "] Elapsed time = " << time << " ms" << endl;
  }
}


GRAPHICS_MAIN()
{
  try
  {
    std::vector<shakti::Device> devices{ shakti::get_devices() };
    cout << devices.back() << endl;

    VideoStream video_stream{ "/home/david/Desktop/GitHub/DO-CV/sara/examples/VideoIO/orion_1.mpg" };
    int video_frame_index{ 0 };
    Image<Rgb8> video_frame;
    Image<float> in_frame;
    Image<float> out_frame;

    while (video_stream.read(video_frame))
    {
      cout << "[Read frame] " << video_frame_index << "" << endl;
      if (!active_window())
        create_window(video_frame.sizes());

      tic();
      in_frame = video_frame.convert<float>();
      out_frame.resize(in_frame.sizes());
      toc("Color conversion");

      tic();
      shakti::gradient(out_frame.data(), in_frame.data(), in_frame.sizes().data());
      toc("Gradient");

      tic();
      display(out_frame);
      toc("Display");

      ++video_frame_index;
      //cout << endl;
    }

  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}
