#include <boost/python.hpp>

#include <DO/Shakti/ImageProcessing.hpp>

#include "ImageProcessing.hpp"
#include "Numpy.hpp"


namespace shakti = DO::Shakti;


template <typename T>
inline T * get_ndarray_data(boost::python::object object)
{
  auto numpy_array = reinterpret_cast<PyArrayObject *>(
    PyArray_FROM_O(object.ptr()));
  return reinterpret_cast<T *>(PyArray_DATA(numpy_array));
}

shakti::Vector2i get_ndarray_shape_2d(boost::python::object object)
{
  auto numpy_array = reinterpret_cast<PyArrayObject *>(
    PyArray_FROM_O(object.ptr()));
  auto shape_data = PyArray_SHAPE(numpy_array);
  const auto w = int(shape_data[1]);
  const auto h = int(shape_data[0]);
  return shakti::Vector2i{ w, h };
}

shakti::Vector3i get_ndarray_shape_3d(boost::python::object object)
{
  auto numpy_array = reinterpret_cast<PyArrayObject *>(
    PyArray_FROM_O(object.ptr()));
  auto shape_data = PyArray_SHAPE(numpy_array);
  const auto w = int(shape_data[1]);
  const auto h = int(shape_data[0]);
  const auto d = int(shape_data[2]);
  return shakti::Vector3i{ w, h, d };
}


void compute_laplacian(boost::python::object out,
                       boost::python::object in)
{
  auto out_data = get_ndarray_data<float>(out);
  const auto in_data = get_ndarray_data<float>(in);
  const auto sizes = get_ndarray_shape_2d(in);
  shakti::compute_laplacian(out_data, in_data, sizes.data());
}

void compute_color_distribution(boost::python::object out,
                                boost::python::object in,
                                int quantization_step)
{
  auto out_data = get_ndarray_data<float>(out);
  const auto in_data = get_ndarray_data<shakti::Vector4ub>(in);
  const auto in_sizes = get_ndarray_shape_2d(in);

  shakti::compute_color_distribution(out_data, in_data, in_sizes.data(),
                                     quantization_step);
}


void expose_imageprocessing()
{
  using namespace boost::python;

  // Import numpy array.
  numeric::array::set_module_and_type("numpy", "ndarray");
  import_numpy_array();

  def("compute_laplacian", &::compute_laplacian);
  def("compute_color_distribution", &::compute_color_distribution);
}
