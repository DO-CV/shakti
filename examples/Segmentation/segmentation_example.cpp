// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <functional>
#include <random>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Segmentation.hpp>
#include <DO/Shakti/Utilities/DeviceInfo.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;

using namespace std;
using namespace sara;


void draw_grid(const Vector2i& sizes, const Vector2i block_sizes)
{
  const auto& w = sizes.x();
  const auto& h = sizes.y();

  Vector2i steps = ((sizes.array() + block_sizes.array() - 1) / block_sizes.array()).matrix();

  cout << sizes.transpose() << endl;
  cout << block_sizes.transpose() << endl;
  cout << steps.transpose() << endl;

  for (auto x = 0; x < steps.x(); ++x)
    draw_line(Point2i{ x*block_sizes.x(), 0 }, Point2i{ x*block_sizes.x(), h }, Green8, 1);

  for (auto y = 0; y < steps.y(); ++y)
    draw_line(Point2i{ 0, y*block_sizes.y() }, Point2i{ w, y*block_sizes.y() }, Green8, 1);

  for (auto y = 0; y < steps.y(); ++y)
    for (auto x = 0; x < steps.x(); ++x)
    {
      fill_circle(
        Point2i{ x*block_sizes.x() + block_sizes.x() / 2,
                 y*block_sizes.y() + block_sizes.y() / 2 },
        3, Green8);
    }
}

void demo_on_image()
{
  auto image_path = src_path("examples/Segmentation/Kingfisher.jpg");
  auto image = Image<Rgba8>{};
  if (!imread(image, image_path))

  {
    cout << "Cannot read image:\n" << image_path << endl;
    return;
  }

  create_window(image.sizes());
  display(image);
  draw_grid(image.sizes(), Vector2i{ 32, 32 });
  //get_key();

  Image<shakti::Vector4f, 2> rgba32f_image{ image.sizes() };
  auto rgba = image.begin();
  auto rgba32f = rgba32f_image.begin();
  for (; rgba != image.end(); ++rgba, ++rgba32f)
  {
    for (int c = 0; c < 4; ++c)
      (*rgba32f)[c] = (*rgba)[c] / 255.f;
    (*rgba32f)[3] = 0.f;
  }

  sara::Timer t;
  t.restart();
  Image<int> labels{ image.sizes() };
  DO::Shakti::SegmentationSLIC slic;
  slic.set_distance_weight(1e-4f);
  slic(labels.data(), rgba32f_image.data(), rgba32f_image.sizes().data());
  cout << "Segmentation time = " << t.elapsed_ms() << "ms" << endl;

  Image<Rgba8> segmentation{ labels.sizes() };
  vector<Rgba64f> means64f(labels.array().maxCoeff()+1, Rgba64f::Zero());
  vector<int> cardinality(labels.array().maxCoeff()+1, 0);

  for (int y = 0; y < segmentation.height(); ++y)
    for (int x = 0; x < segmentation.width(); ++x)
    {
      Rgba64f col;
      smart_convert_color(image(x, y), col);
      means64f[labels(x, y)] += col;
      ++cardinality[labels(x, y)];
    }

  std::vector<Rgba8> means(means64f.size());
  for (int i = 0; i < means64f.size(); ++i)
  {
    means64f[i] /= cardinality[i];
    smart_convert_color(means64f[i], means[i]);
  }

  for (int y = 0; y < segmentation.height(); ++y)
    for (int x = 0; x < segmentation.width(); ++x)
      segmentation(x, y) = means[labels(x, y)];

  display(segmentation);
  get_key();
}

void demo_on_video()
{
  auto devices = shakti::get_devices();
  devices.front().make_current_device();
  cout << devices.front() << endl;

  VideoStream video_stream{
#ifdef _WIN32
    "C:/Users/David/Desktop/GitHub/sara/examples/VideoIO/orion_1.mpg"
#else
    "/home/david/Desktop/GitHub/DO-CV/sara/examples/VideoIO/orion_1.mpg"
#endif
  };
  auto video_frame_index = int{ 0 };
  auto video_frame = Image<Rgb8>{};

  while (video_stream.read(video_frame))
  {
    cout << "[Read frame] " << video_frame_index << "" << endl;

    video_frame = enlarge(video_frame, Vector2i{ 640, 480 });
    if (!active_window())
    {
      create_window(video_frame.sizes());
    }

    Image<shakti::Vector4f, 2> rgba32f_image{ video_frame.sizes() };
    auto rgba = video_frame.begin();
    auto rgba32f = rgba32f_image.begin();
    for (; rgba != video_frame.end(); ++rgba, ++rgba32f)
    {
      for (int i = 0; i < 3; ++i)
        (*rgba32f)[i] = (*rgba)[i] / 255.f;
      (*rgba32f)[3] = 0.f;
    }

    sara::Timer t;
    t.restart();
    Image<int> labels{ video_frame.sizes() };
    DO::Shakti::SegmentationSLIC slic;
    slic.set_distance_weight(1e-4f);
    slic(labels.data(), rgba32f_image.data(), rgba32f_image.sizes().data());
    cout << "Segmentation time = " << t.elapsed_ms() << "ms" << endl;

    Image<Rgba8> segmentation{ labels.sizes() };
    vector<Rgba64f> means64f(labels.array().maxCoeff() + 1, Rgba64f::Zero());
    vector<int> cardinality(labels.array().maxCoeff() + 1, 0);

    for (int y = 0; y < segmentation.height(); ++y)
      for (int x = 0; x < segmentation.width(); ++x)
      {
        Rgba64f col;
        smart_convert_color(video_frame(x, y), col);
        means64f[labels(x, y)] += col;
        ++cardinality[labels(x, y)];
      }

    std::vector<Rgba8> means(means64f.size());
    for (int i = 0; i < means64f.size(); ++i)
    {
      means64f[i] /= cardinality[i];
      smart_convert_color(means64f[i], means[i]);
    }

    for (int y = 0; y < segmentation.height(); ++y)
      for (int x = 0; x < segmentation.width(); ++x)
        segmentation(x, y) = means[labels(x, y)];

    display(segmentation);

    ++video_frame_index;
    cout << endl;
  }
}

GRAPHICS_MAIN()
{
  try
  {
    demo_on_video();
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }

  return 0;
}