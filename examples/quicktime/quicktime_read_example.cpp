#include <memory>
#include <vector>

#include <lqt/lqt.h>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include "../image_processing/image_processing.hpp"


namespace DO { namespace Sara {

  class QuicktimeVideoStream
  {
  public:
    QuicktimeVideoStream() = default;

    QuicktimeVideoStream(const std::string& video_filepath)
    {
      _video_file = quicktime_open(video_filepath.c_str(), 1, 0);
      _num_tracks = quicktime_video_tracks(_video_file);

      _current_track = 0;
      _current_track_supported = false;
    }

    ~QuicktimeVideoStream()
    {
      quicktime_close(_video_file);
    }

    Vector2i sizes() const
    {
      return Vector2i{
        quicktime_video_width(_video_file, _current_track),
        quicktime_video_height(_video_file, _current_track)
      };
    }

    int depth() const
    {
      return quicktime_video_depth(_video_file, _current_track);
    }

    std::string color_model() const
    {
      auto color_model = lqt_get_cmodel(_video_file, _current_track);
      auto color_model_name = lqt_colormodel_to_string(color_model);
      return color_model_name;
    }

    int current_timestamp() const
    {
      return lqt_frame_time(_video_file, _current_track);
    }

    double current_frame_rate() const
    {
      return
        static_cast<double>(lqt_video_time_scale(_video_file, _current_track)) /
        lqt_frame_duration(_video_file, _current_track, nullptr);
    }

    void bind_frame_rows(Image<Rgb8>& frame)
    {
      auto sizes = this->sizes();
      frame.resize(sizes);
      _current_frame_rows.resize(sizes[1]);
      for (int y = 0; y < sizes[1]; ++y)
        _current_frame_rows[y] = reinterpret_cast<unsigned char *>(&frame(0,y));
    }

    bool read(Image<Rgb8>& video_frame, bool bind_frame = true)
    {
      if (!_current_track_supported &&
          quicktime_supported_video(_video_file, _current_track) == 0)
      {
        std::cout
          << "Movie track " << _current_track
          << " is unsupported by liquicktime!" << std::endl;
        return false;
      }

      if (video_frame.sizes() != sizes() || bind_frame)
        bind_frame_rows(video_frame);
      quicktime_decode_video(_video_file, _current_frame_rows.data(), _current_track);

      return true;
    }

  private:
    quicktime_t *_video_file;
    int _num_tracks;
    std::vector<unsigned char *> _video_rows;

    int _current_track;
    bool _current_track_supported;
    std::vector<unsigned char *> _current_frame_rows;
  };

} /* namespace Sara */
} /* namespace DO */


namespace {

  static DO::Sara::Timer timer;

  void tic()
  {
    timer.restart();
  }

  void toc(const char *what)
  {
    auto time = timer.elapsedMs();
    std::cout << "[" << what << "] Elapsed time = " << time << " ms" << std::endl;
  }
}


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


using namespace std;


template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


GRAPHICS_MAIN()
{
  auto video_filepath = string{ "/home/david/Desktop/HAVAS_DANONE_PITCH_EP_1011.mov" };
  auto video_stream = sara::QuicktimeVideoStream{ video_filepath };
  auto video_frame = sara::Image<sara::Rgb8>{};

  auto in_frame = sara::Image<float>{};
  auto out_frame = sara::Image<float>{};

  video_stream.bind_frame_rows(video_frame);

  int frame = 0;
  while (true)
  {
    video_stream.read(video_frame, false);

    tic();
    in_frame = video_frame.convert<float>();
    out_frame.resize(in_frame.sizes());
    toc("Color conversion");

    tic();
    shakti::gradient(out_frame.data(), in_frame.data(), in_frame.sizes().data());
    //out_frame = in_frame.compute<sara::Gradient>().compute<sara::SquaredNorm>();
    //out_frame.array() = out_frame.array().sqrt();
    toc("Gradient");

    tic();
    if (!sara::active_window())
      sara::create_window(video_frame.sizes());
    sara::display(out_frame);
    toc("Display");

    ++frame;
  }

  return 0;
}
