set -e

# Create the build directory.
if [ -d "../shakti-build-shared" ]; then
  rm -rf ../shakti-build-shared
fi
mkdir ../shakti-build-shared
cd ../shakti-build-shared

# Create the CMake project.
cmake ../shakti \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.5 \
  -DSHAKTI_BUILD_SHARED_LIBS=ON \
  -DSHAKTI_BUILD_PYTHON_BINDINGS=ON \
  -DSHAKTI_BUILD_TESTS=ON \
  -DSHAKTI_BUILD_SAMPLES=ON

# Build the library.
make -j`nproc` && make test && make package

# Register package to local debian repository.
dpkg-sig --sign builder ../shakti-build-shared/libDO-Shakti-shared-*.deb
sudo cp ../shakti-build-shared/libDO-Shakti-shared-*.deb /usr/local/debs
sudo update-local-debs
sudo apt-get update
sudo apt-get install --reinstall libdo-shakti-shared
