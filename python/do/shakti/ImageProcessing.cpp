#include <boost/python.hpp>

#include <DO/Shakti/ImageProcessing.hpp>

#include "ImageProcessing.hpp"
#include "Numpy.hpp"


namespace shakti = DO::Shakti;


void compute_laplacian(PyObject *in, PyObject *out)
{
  const auto in_data = shakti::get_ndarray_data<float>(in);
  const auto sizes = shakti::get_ndarray_shape_2d(in);
  auto out_data = shakti::get_ndarray_data<float>(out);

  shakti::compute_laplacian(out_data, in_data, sizes.data());
}

void compute_color_distribution(PyObject *in, PyObject *out,
                                int quantization_step)
{
  const auto in_data = shakti::get_ndarray_data<shakti::Vector4ub>(in);
  const auto in_sizes = shakti::get_ndarray_shape_2d(in);
  auto out_data = shakti::get_ndarray_data<float>(out);

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
