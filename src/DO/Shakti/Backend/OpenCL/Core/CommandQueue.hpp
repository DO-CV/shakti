#ifndef DO_SHAKTI_BACKEND_OPENCL_CORE_COMMANDQUEUE_HPP
#define DO_SHAKTI_BACKEND_OPENCL_CORE_COMMANDQUEUE_HPP

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include <DO/Shakti/Backend/OpenCL/Core/Context.hpp>
#include <DO/Shakti/Backend/OpenCL/Core/Device.hpp>
#include <DO/Shakti/Backend/OpenCL/Core/DeviceBuffer.hpp>
#include <DO/Shakti/Backend/OpenCL/Core/DeviceImage.hpp>
#include <DO/Shakti/Backend/OpenCL/Core/Error.hpp>
#include <DO/Shakti/Backend/OpenCL/Core/Kernel.hpp>


namespace DO { namespace Shakti { namespace OpenCL {

  class CommandQueue
  {
  public:
    CommandQueue() = default;

    CommandQueue(const Context& context, const Device& device)
    {
      if (!initialize(context, device))
        throw std::runtime_error{ "Error: failed to initialize command queue!" };
    }

    ~CommandQueue()
    {
      release();
    }

    bool initialize(const Context& context, const Device& device)
    {
      auto err = cl_int{};
      _queue = clCreateCommandQueue(context, device, _properties, &err);

      if (err < 0)
      {
        std::cerr << format("Error: failed to create command queue! %s",
                            get_error_string(err)) << std::endl;
        return false;
      }

      return true;
    }

    bool release()
    {
      auto err = clReleaseCommandQueue(_queue);
      if (err < 0)
      {
        std::cerr << format("Error: failed to release command queue! %s",
                            get_error_string(err)) << std::endl;
        return false;
      }
      return true;
    }

    bool finish()
    {
      auto err = clFinish(_queue);
      if (err < 0)
      {
        std::cerr << format("Error: failed to finish command queue! %s",
                            get_error_string(err)) << std::endl;
        return false;
      }
      return true;
    }

    bool enqueue_nd_range_kernel(Kernel& kernel,
                                 cl_uint work_dims,
                                 const size_t *global_work_offsets,
                                 const size_t *global_work_sizes,
                                 const size_t *local_work_sizes)
    {
      auto err = clEnqueueNDRangeKernel(_queue, kernel, work_dims,
                                        global_work_offsets, global_work_sizes,
                                        local_work_sizes, 0, nullptr, nullptr);

      if (err)
      {
        std::cerr << format("Error: Failed to execute kernel! %s",
                            get_error_string(err)) << std::endl;
        return false;
      }

      return true;
    }

    template <typename T>
    bool enqueue_read_buffer(Buffer<T>& src, T *dst, bool blocking = true)
    {
      auto err = clEnqueueReadBuffer(_queue, src, cl_bool(blocking), 0,
                                     src.size() * sizeof(T), dst, 0, nullptr,
                                     nullptr);

      if (err)
      {
        std::cerr << format("Error: Failed to copy buffer from device to host! "
                            "%s",
                            get_error_string(err)) << std::endl;
        return false;
      }

      return true;
    }

    template <typename T>
    bool enqueue_read_image(Image<T, 2>& src, T *dst, bool blocking = true)
    {
      size_t origin[3] = { 0, 0, 0 };
      size_t region[3] = { src.width(), src.height(), 1 };

      auto err = clEnqueueReadImage(_queue, src, cl_bool(blocking), origin,
                                    region, 0, 0, dst, 0, nullptr, nullptr);
      if (err)
      {
        std::cerr << format("Error: Failed to copy buffer from device to host! "
                            "%s",
                            get_error_string(err)) << std::endl;
        return false;
      }

      return true;
    }

    operator cl_command_queue() const
    {
      return _queue;
    }

  private:
    cl_command_queue _queue{ nullptr };
    cl_command_queue_properties _properties{ 0 };
  };

} /* namespace OpenCL */
} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_BACKEND_OPENCL_CORE_COMMANDQUEUE_HPP */
