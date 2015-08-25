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


using namespace std;
using namespace DO::Shakti;


TEST(TestMatrix, test_constructors)
{
  Vector2i p{ 2, 1 };
  EXPECT_EQ(p.x(), 2);
  EXPECT_EQ(p.y(), 1);

  const int sizes[] = { 1, 3 };
  p = sizes;
  EXPECT_EQ(p.x(), sizes[0]);
  EXPECT_EQ(p.y(), sizes[1]);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}