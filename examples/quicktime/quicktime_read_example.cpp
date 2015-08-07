#include <memory>
#include <vector>

#include <lqt/lqt.h>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Graphics.hpp>


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

    bool read(Image<Rgb8>& video_frame);

  private:
    quicktime_t *_video_file;
    int _num_tracks;
    std::vector<unsigned char *> _video_rows;

    int _current_track;
    Image<Rgb8> _current_frame;
    std::vector<unsigned char *> _current_frame_rows;
  };

} /* namespace Sara */
} /* namespace DO */


namespace sara = DO::Sara;


using namespace std;



template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


GRAPHICS_MAIN()
{
  auto movie_filepath = string{ "/home/david/Downloads/sample_iTunes.mov" };
  auto movie_file = quicktime_open(movie_filepath.c_str(), 1, 0);

  auto movie_tracks = quicktime_video_tracks(movie_file);
  CHECK(movie_tracks);

  for (int track = 0; track < movie_tracks; ++track)
  {
    if (quicktime_supported_video(movie_file, track) == 0)
    {
      cout << "Movie track " << track << " is unsupported by liquicktime!" << endl;
      continue;
    }

    auto video_frame_sizes = sara::Vector2i{
      quicktime_video_width(movie_file, track),
      quicktime_video_height(movie_file, track)
    };
    auto video_depth = quicktime_video_depth(movie_file, track);

    auto video_frame = sara::Image<sara::Rgb8>{ video_frame_sizes };
    auto video_rows = vector<unsigned char *>{ video_frame_sizes.y() };
    for (int r = 0; r < video_frame_sizes[1]; ++r)
      video_rows[r] = reinterpret_cast<unsigned char *>(&video_frame(0, r));

    double video_frame_rate = lqt_video_time_scale(movie_file, track) / lqt_frame_duration(movie_file, track, nullptr);
    auto color_model = lqt_get_cmodel(movie_file, track);
    auto color_model_name = lqt_colormodel_to_string(color_model);
    CHECK(color_model_name);

    auto previous_timestamp = int64_t{ -1 };
    auto timestamp = int64_t{};
    auto ret = int{};
    do
    {
      timestamp = lqt_frame_time(movie_file, track);
      ret = lqt_decode_video(movie_file, video_rows.data(), track);

      CHECK(previous_timestamp);
      CHECK(timestamp);
      CHECK(ret);

      if (!sara::active_window())
        sara::create_window(video_frame_sizes);
      sara::display(video_frame);

      previous_timestamp = timestamp;
    } while(ret == 0);
  }

  quicktime_close(movie_file);

  return 0;
}
