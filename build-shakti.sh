mkdir ../shakti-build-shared
cd ../shakti-build-shared
cmake ../shakti \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.5 \
  -DSHAKTI_BUILD_SHARED_LIBS=ON \
  -DSHAKTI_BUILD_PYTHON_BINDINGS=ON \
  -DSHAKTI_BUILD_TESTS=ON \
  -DSHAKTI_BUILD_SAMPLES=ON

make -j`nproc` && make test && make package
