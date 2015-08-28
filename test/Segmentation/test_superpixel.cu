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

#include <DO/Shakti/ImageProcessing/Kernels/Globals.hpp>

#include <DO/Shakti/MultiArray.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>


using namespace std;
using namespace DO::Shakti;


TEST(TestSegmentation, test_me)
{
  EXPECT_FALSE(true);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
