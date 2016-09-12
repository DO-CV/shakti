#ifndef DO_SHAKTI_BACKEND_OPENCL_CORE_DEVICEBUFFER_HPP
#define DO_SHAKTI_BACKEND_OPENCL_CORE_DEVICEBUFFER_HPP

#include <DO/Shakti/Backend/OpenCL/Core/Program.hpp>


namespace DO { namespace Shakti { namespace OpenCL {

  template <typename T>
  class Buffer
  {
  public:
    inline Buffer() = default;

    inline Buffer(Context& context, T *data, size_t size,
                  cl_mem_flags flags = CL_MEM_READ_WRITE)
      : _data(data)
      , _size(size)
      , _buffer(nullptr)
    {
      auto err = cl_int{};
      _buffer = clCreateBuffer(context, flags, size * sizeof(T), _data, &err);
      if (err < 0)
        throw std::runtime_error(
            format("Error: failed to allocate buffer in device memory! %s\n",
                   get_error_string(err)));
    }

    inline ~Buffer()
    {
      auto err = clReleaseMemObject(_buffer);
      if (err < 0)
        std::cerr << format(
                         "Error: failed to release buffer in device memory! %s",
                         get_error_string(err)) << std::endl;
    }

    operator cl_mem&()
    {
      return _buffer;
    }

    size_t size() const
    {
      return _size;
    }

  private:
    T *_data{ nullptr };
    size_t _size{ 0 };
    cl_mem _buffer{ nullptr };
  };

} /* namespace kojo */


#endif /* DO_SHAKTI_BACKEND_OPENCL_CORE_DEVICEBUFFER_HPP */
