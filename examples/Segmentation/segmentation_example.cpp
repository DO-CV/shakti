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


#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>


using namespace std;
using namespace DO::Sara;

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

GRAPHICS_MAIN()
{
  auto image_path = src_path("examples/Segmentation/Kingfisher.jpg");
  auto image = Image<Rgba8>{};
  if (!imread(image, image_path))
  {
    cout << "Cannot read image:\n" << image_path << endl;
    return -1;
  }

  create_window(image.sizes());
  display(image);
  draw_grid(image.sizes(), Vector2i{ 32, 32 });
  get_key();

  return 0;
}