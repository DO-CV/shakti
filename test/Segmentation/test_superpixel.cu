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

#include <memory>

#include <gtest/gtest.h>

#include <DO/Shakti/Segmentation/SuperPixel.hpp>


using namespace std;
using namespace DO::Shakti;


TEST(TestSegmentationSLIC, test_getters_and_setters)
{
  SegmentationSLIC slic;

  slic.set_image_sizes({ 2, 2 });
  EXPECT_EQ(Vector2i(2, 2), slic.get_image_sizes());

  slic.set_image_padded_width(128);
  EXPECT_EQ(128, slic.get_image_padded_width());
}



TEST(TestSegmentation, test_algorithm)
{
  // Test host image data.
  const int N = 3;
  const int B = 16;
  const int M = N*B;
  const auto host_image_sizes = Vector2i{ N, N };
  std::unique_ptr<Vector3f[]> in_host_image(new Vector3f[M*M]);

  auto at = [&](int x, int y) { return x + M*y; };
  for (int y = 0; y < M; ++y)
  {
    for (int x = 0; x < M; ++x)
    {
      auto val = float(x / B + N * (y / B)) / (N*N);
      in_host_image[at(x, y)] = Vector3f{ val, val, val };
    }
  }

  // Transfer data to device memory.
  MultiArray<Vector3f, 2> in_device_image{ in_host_image.get(), host_image_sizes };

  SegmentationSLIC slic;

  // Test helper function for image dimensions.
  slic.set_image_sizes(in_device_image);
  EXPECT_EQ(in_device_image.sizes(), slic.get_image_sizes());
  EXPECT_EQ(in_device_image.padded_width(), slic.get_image_padded_width());

  // Now test the algorithm itself.
  MultiArray<int, 2> out_device_labels{ slic(in_device_image) };

  std::unique_ptr<int[]> in_host_labels(new int[M*M]);
  for (int y = 0; y < M; ++y)
  {
    for (int x = 0; x < M; ++x)
      cout << in_host_labels[at(x, y)] << " ";
    cout << endl;
  }
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
