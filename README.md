Shakti: CUDA-accelerated Computer Vision Algorithms
===================================================

Shakti focuses on:

1. having an **easy-to-use and simple API**,
2. having **easy-to-understand and efficient** implementations of computer vision
   algorithms,
3. **rigorous testing**.


Shakti is licensed with the [Mozilla Public License version 2.0](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2).


Some parts of the Shakti depend on Sara (https://github.com/DO-CV/Sara).

Shakti provides the following classical algorithms:
- Simple matrix data structures that are compatible with CUDA as Eigen does not play well with CUDA yet;
- ND-array data structures that wraps parts of the CUDA API for ease of reuse;
- Basic linear filtering (gradient, laplacian, gaussian convolution, separable convolution);
- Dense SIFT;
- Superpixel segmentation
