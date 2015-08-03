#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Utilities/DeviceInfo.hpp>

#include "toy_test_cuda.h"


using namespace std;

using namespace DO;
using namespace DO::Sara;


__device__
void test()
{
}


GRAPHICS_MAIN()
{
  try
  {
    std::vector<Shakti::Device> devices{Shakti::get_devices()};
    cout << devices.back() << endl;
    toy_test_cuda();

    create_window(200, 200, "Hello Shakti!");
    get_key();
  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}
