#ifndef DO_SHAKTI_BACKEND_OPENCL_CORE_CONTEXT_HPP
#define DO_SHAKTI_BACKEND_OPENCL_CORE_CONTEXT_HPP

#include <vector>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include <DO/Sara/Core/StringFormat.hpp>

#include <DO/Shakti/Backend/OpenCL/Core/Device.hpp>


namespace DO { namespace Shakti { namespace OpenCL {

  class Context
  {
  public:
    Context(const Device& device)
    {
      auto err = cl_int{};
      _context = clCreateContext(nullptr, 1, &device.id, nullptr, nullptr, &err);
      if (err < 0)
        throw std::runtime_error{ format(
            "Error: failed to create context from device: %d! %s\n", device.id,
            get_error_string(err))
        };

      err = clGetContextInfo(_context, CL_CONTEXT_REFERENCE_COUNT,
                             sizeof(_ref_count), &_ref_count, nullptr);
    }

    ~Context()
    {
      auto err = clReleaseContext(_context);
      if (err < 0)
        std::cerr << format("Error: failed to release OpenCL program! %s\n",
                            get_error_string(err)) << std::endl;
    }

    operator cl_context() const
    {
      return _context;
    }

    template <typename T>
    void push_property(cl_uint key, T value)
    {
      _properties.push_back(key);
      _properties.push_back(reinterpret_cast<cl_context_properties>(value));
    }

  private:
    cl_context _context{ nullptr };
    cl_uint _ref_count{ 0 };
    std::vector<cl_context_properties> _properties;
  };

} /* namespace OpenCL */
} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_BACKEND_OPENCL_CORE_CONTEXT_HPP */
