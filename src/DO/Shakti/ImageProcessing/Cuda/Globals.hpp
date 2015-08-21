#ifndef DO_SHAKTI_IMAGEPROCESSING_CUDA_GLOBALS_HPP
#define DO_SHAKTI_IMAGEPROCESSING_CUDA_GLOBALS_HPP


namespace DO { namespace Shakti {

  texture<float, 2> in_float_texture;
  texture<float2, 2> in_float2_texture;

  __constant__ float convolution_kernel[1024];
  __constant__ int convolution_kernel_size;

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_CUDA_GLOBALS_HPP */