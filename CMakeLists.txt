cmake_minimum_required(VERSION 2.8.12)

project(Shakti)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

option(DO_BUILD_SAMPLES "Build samples in Shakti libraries" ON)
option(DO_BUILD_TESTS "Build unit tests for Shakti libraries" ON)

# Set CMake module path.
if (WIN32)
  set(DO_Sara_DIR "C:/Program Files/DO-Sara/include/DO/Sara")
else ()
  set(DO_Sara_DIR /usr/include/DO/Sara)
endif ()
list(APPEND CMAKE_MODULE_PATH
     ${DO_Sara_DIR}/cmake
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find third-party libraries.
find_package(CUDA 7.0 REQUIRED)
find_package(DO_Sara COMPONENTS Core Graphics ImageIO VideoIO REQUIRED)


# Use custom CMake macros.
include(do_convenience_macros)
include(do_configure_nvcc_compiler)

include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${DO_Sara_INCLUDE_DIRS}
  src
  third_party/gtest/include
)

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# Build the following directories.
add_subdirectory(third_party)
add_subdirectory(src)

if (DO_BUILD_TESTS)
  add_subdirectory(test)
endif ()

if (DO_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()
