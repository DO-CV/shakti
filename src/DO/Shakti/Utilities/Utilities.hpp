#ifndef DO_SHAKTI_UTILITIES_UTILITIES_HPP
#define DO_SHAKTI_UTILITIES_UTILITIES_HPP


namespace DO { namespace Shakti{

  template <typename T, int N, enum TextureReadMode readMode>
  inline void bind_texture(const texture<T, N, ReadMode>& tex,
                           const cudaArray *cuda_array)
  {
    CHECK_CUDA_RUNTIME_ERROR(cudaBindTextureToArray(tex, cuda_array));
  }

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_UTILITIES_UTILITIES_HPP */