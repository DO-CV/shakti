#pragma once

#include <DO/Shakti/MultiArray.hpp>


namespace DO { namespace Shakti {

  void gradient(float *out, const float *in, const int *sizes);

} /* namespace Shakti */
} /* namespace DO */