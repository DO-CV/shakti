#include <boost/python.hpp>

#include "ImageProcessing.hpp"


BOOST_PYTHON_MODULE(shakti)
{
  expose_imageprocessing();
}
