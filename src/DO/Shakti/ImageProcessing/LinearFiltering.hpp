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

#ifndef DO_SHAKTI_IMAGEPROCESSING_LINEARFILTERING_HPP
#define DO_SHAKTI_IMAGEPROCESSING_LINEARFILTERING_HPP

#include <vector>

#include <DO/Shakti/MultiArray/MultiArray.hpp>
#include <DO/Shakti/MultiArray/TextureArray.hpp>


namespace DO { namespace Shakti {

  void apply_row_based_convolution(
    float *out, const float *in, const float *kernel,
    int kernel_size, const int *sizes);

  void apply_column_based_convolution(
    float *out, const float *in, const float *kernel,
    int kernel_size, const int *sizes);

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  class GaussianFilter
  {
  public:
    GaussianFilter(float sigma, int truncation_factor = 4.f)
      : _truncation_factor{ truncation_factor }
    {
      set_sigma(sigma);
    }

    void set_sigma(float sigma);

    void operator()(float *out, const float *in, const int *sizes) const;

    MultiArray<float, 2> operator()(TextureArray<float>& in) const;

  private:
    int _truncation_factor;
    std::vector<float> _kernel;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_LINEARFILTERING_HPP */