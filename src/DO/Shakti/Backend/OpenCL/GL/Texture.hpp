#ifndef DO_SHAKTI_OPENCL_GL_TEXTURE_HPP
#define DO_SHAKTI_OPENCL_GL_TEXTURE_HPP

#include <gl/glew.h>


namespace kojo { namespace gl {

  template <typename T>
  struct PixelTraits;


  template <>
  struct PixelTraits < float >
  {
    enum {
      PixelType = GL_FLOAT,
      PixelFormat = GL_INTENSITY32F_ARB,
      ColorSpace = GL_INTENSITY
    };
  };


  class Texture2D
  {
  public:
    Texture2D()
    {
      glGenTextures(1, &_tex_id);
    }

    ~Texture2D()
    {
      glBindTexture(1, _tex_id);
      glDeleteTextures(1, &_tex_id);
    }

    inline operator GLuint() const
    {
      return _tex_id;
    }

    void bind() const
    {
      glBindTexture(GL_TEXTURE_2D, _tex_id);
    }

    void unbind() const
    {
      glBindTexture(GL_TEXTURE_2D, 0);
    }

    template <typename T>
    void upload(const PixelBuffer<T>& pixel_buffer,
                int level = 0,
                int border_type = GL_CLAMP_TO_EDGE)
    {
      glTexImage2D(
        GL_TEXTURE_2D, level, PixelTraits<T>::PixelFormat,
        pixel_buffer.width(), pixel_buffer.height(),
        GL_CLAMP_TO_EDGE,
        PixelTraits<T>::ColorSpace, PixelTraits<T>::PixelType);
    }

  private:
    GLuint _tex_id;
  };

} /* namespace GL */
} /* namespace OpenCL */
} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_OPENCL_GL_TEXTURE_HPP */
