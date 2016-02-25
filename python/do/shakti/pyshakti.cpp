#include <boost/python.hpp>

#include "ImageProcessing.hpp"


BOOST_PYTHON_MODULE(pyshakti)
{
  expose_imageprocessing();
}
