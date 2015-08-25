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

#include <gtest/gtest.h>

#include <DO/Shakti/MultiArray.hpp>


namespace shakti = DO::Shakti;

using namespace std;


TEST(TestMultiArray, test_constructors)
{
  shakti::MultiArray<float, 2> matrix{ { 3, 4 } };
  EXPECT_EQ(matrix.sizes(), shakti::Vector2i(3, 4));
  EXPECT_EQ(matrix.size(0), 3);
  EXPECT_EQ(matrix.size(1), 4);
}

TEST(MultiArray, test_copy_between_host_and_device)
{
  const int w = 3;
  const int h = 4;
  float in_host_data[] = {
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11
  };

  // Copy to device.
  shakti::MultiArray<float, 2> out_device_image{ in_host_data, { w, h } };

  // Copy back to host.
  float out_host_data[w*h];
  out_device_image.copy_to_host(out_host_data);

  EXPECT_TRUE(std::equal(in_host_data, in_host_data + w*h, out_host_data));
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}