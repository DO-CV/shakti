set -e

# Create the build directory.
if [ -d "../shakti-build" ]; then
  rm -rf ../shakti-build
fi
mkdir ../shakti-build

cd ../shakti-build
{
  # Create the CMake project.
  cmake ../shakti \
    -DCMAKE_BUILD_TYPE=Release \
    -DSHAKTI_BUILD_SHARED_LIBS=ON \
    -DSHAKTI_BUILD_PYTHON_BINDINGS=ON \
    -DSHAKTI_BUILD_TESTS=ON \
    -DSHAKTI_BUILD_SAMPLES=ON

  # Build the library.
  make -j`nproc` && make test && make pytest && make package

  if [ -f "/etc/debian_version" ]; then
    # Register package to local debian repository.
    dpkg-sig --sign builder libDO-Shakti-shared-*.deb
    sudo cp libDO-Shakti-shared-*.deb /usr/local/debs
    sudo update-local-debs
    sudo apt-get update
    sudo apt-get install --reinstall libdo-shakti-shared
  else
    rpm_package_name=$(echo `ls *.rpm`)
    sudo rpm -ivh --force ${rpm_package_name}
  fi
}
cd ..
