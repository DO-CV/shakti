#ifndef DO_SHAKTI_OPENCL_GL_PIXELBUFFER_HPP
#define DO_SHAKTI_OPENCL_GL_PIXELBUFFER_HPP

#include <array>
#include <numeric>

#include <GL/gl.h>
#include <GL/glew.h>


namespace DO { namespace Shakti { namespace OpenCL { namespace GL {

  template <typename T, int N = 2>
  class PixelBuffer
  {
  public:
    typedef T pixel_type;
    enum { Dimension = N };

  public:
    inline PixelBuffer(size_t width, size_t height, const T *data = nullptr)
      : _dims({ width, height })
    {
      glGenBuffers(1, &_pbo_id);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo_id);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(T), data,
                   GL_DYNAMIC_DRAW);
    }

    inline PixelBuffer(size_t width, size_t height, size_t depth,
                       const T *data = nullptr)
      : _dims({ width, height, depth })
    {
      glGenBuffers(1, &_pbo_id);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo_id);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * depth * sizeof(T),
                   data, GL_DYNAMIC_DRAW);
    }

    inline ~PixelBuffer()
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &_pbo_id);
    }

    inline operator GLuint() const
    {
      return _pbo_id;
    }

    inline size_t width() const
    {
      return _dims[0];
    }

    inline size_t height() const
    {
      return _dims[1];
    }

    inline size_t depth() const
    {
      return _dims[2];
    }

    inline size_t byte_size() const
    {
      return sizeof(pixel_type) * std::accumulate(_dims.begin(), _dims.end(),
                                                  static_cast<size_t>(1),
                                                  std::multiplies<size_t>());
    }

    inline void bind() const
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo_id);
    }

    inline void unbind() const
    {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    void unpack(const T *data) const
    {
      glBufferData(GL_PIXEL_UNPACK_BUFFER, byte_size(), data, GL_DYNAMIC_DRAW);
    }

  private:
    GLuint _pbo_id;
    std::array<size_t, N> _dims;
  };


} /* namespace GL */
} /* namespace OpenCL */
} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_OPENCL_GL_PIXELBUFFER_HPP */
