#ifndef DO_SHAKTI_BACKEND_OPENCL_CORE_PLATFORM_HPP
#define DO_SHAKTI_BACKEND_OPENCL_CORE_PLATFORM_HPP

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include <DO/Shakti/Backend/OpenCL/Core/Error.hpp>


namespace DO { namespace Shakti { namespace OpenCL {

  struct Platform
  {
    cl_platform_id id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string profile;
    std::string extensions;

    friend inline std::ostream& operator<<(std::ostream& os, const Platform& p)
    {
      const size_t length = std::string("Extensions  ").size();
      using std::left;
      using std::setw;
      os << left << setw(length) << "ID"         << ":  " << p.id << "\n";
      os << left << setw(length) << "Name"       << ":  " << p.name << "\n";
      os << left << setw(length) << "Vendor"     << ":  " << p.vendor << "\n";
      os << left << setw(length) << "Version"    << ":  " << p.version << "\n";
      os << left << setw(length) << "Profile"    << ":  " << p.profile << "\n";
      os << left << setw(length) << "Extensions" << ":  " << p.extensions << "\n";
      return os;
    }
  };

  template <int _InfoType>
  std::string get_platform_info(cl_platform_id platform_id)
  {
    cl_int err;

    size_t buffer_length;
    err = clGetPlatformInfo(platform_id, _InfoType, 0, nullptr, &buffer_length);
    if (err < 0)
      throw std::runtime_error(format(
      "Error: cannot get platform info! %s",
      get_error_string(err)));

    std::vector<char> buffer;
    buffer.resize(buffer_length);
    err = clGetPlatformInfo(platform_id, _InfoType, buffer_length, &buffer[0],
                            nullptr);
    if (err < 0)
      throw std::runtime_error(format(
      "Error: cannot get platform info! %s",
      get_error_string(err)));

    return std::string(buffer.begin(), buffer.end());
  }


  std::vector<Platform> get_platforms()
  {
    cl_int err;

    cl_uint num_platforms;
    err = clGetPlatformIDs(1, nullptr, &num_platforms);
    if (err < 0)
      throw std::runtime_error(format(
      "Error: cannot get number of platforms! %s", get_error_string(err)));

    std::vector<cl_platform_id> platform_ids(num_platforms);
    err = clGetPlatformIDs(num_platforms, &platform_ids[0], nullptr);
    if (err < 0)
      throw std::runtime_error(format(
      "Error: cannot get the list of platforms",
      get_error_string(err)));

    std::vector<Platform> platforms(num_platforms);
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
      const auto& id = platform_ids[i];
      auto& platform = platforms[i];

      platform.id = id;
      platform.name = get_platform_info<CL_PLATFORM_NAME>(id);
      platform.vendor = get_platform_info<CL_PLATFORM_VENDOR>(id);
      platform.version = get_platform_info<CL_PLATFORM_VERSION>(id);
      platform.profile = get_platform_info<CL_PLATFORM_PROFILE>(id);
      platform.extensions = get_platform_info<CL_PLATFORM_EXTENSIONS>(id);
    }

    return platforms;
  }

} /* namespace OpenCL */
} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_BACKEND_OPENCL_CORE_PLATFORM_HPP */
