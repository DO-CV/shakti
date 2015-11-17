Shakti: CUDA Computer Vision Algorithms
=======================================

*Shakti* (शक्ति) is a lovely female Hindi first name meaning *power*.

*Shakti* focuses on:

1. having an **easy-to-use and simple API**,
2. having **easy-to-understand and efficient** implementations of computer vision
   algorithms,
3. **rigorous testing**.


*Shakti* is licensed with the [Mozilla Public License version 2.0](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2).


*Shakti* depend on *Sara* (https://github.com/DO-CV/Sara).

*Shakti* provides the following classical algorithms:
- Simple matrix data structures that are compatible with CUDA as Eigen does not play well with CUDA yet;
- ND-array data structures that wraps parts of the CUDA API for ease of reuse;
- Basic linear filtering (gradient, laplacian, gaussian convolution, separable convolution);
- Dense SIFT;
- Superpixel segmentation


Build the libraries
-------------------

Make sure that **CUDA >= 7.0** and **Sara** is installed.

1. Install the following packages:

   ```
   mkdir build
   cd build
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DSHAKTI_BUILD_SHARED_LIBS=ON \
     -DSHAKTI_BUILD_SAMPLES=ON \
     -DSHAKTI_BUILD_TESTS=ON
   make  -j`nproc`  # to build with all your CPU cores.
   ```

3. Run the tests to make sure everything is alright.

   ```
   ctest --output-on-failure
   ```

4. Create DEB and RPM package.

   ```
   make package
   ```

5. Deploy by install the Debian package with Ubuntu Software Center, or type:

   ```
   # Debian-based distros:
   sudo dpkg -i libDO-Shakti-shared-{version}.deb

   # Red Hat-based distros:
   sudo rpm -i libDO-Shakti-shared-{version}.deb
   ```
