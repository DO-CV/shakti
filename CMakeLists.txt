cmake_minimum_required(VERSION 2.8.12)

set(DO_Shakti_VERSION_MAJOR 0)
set(DO_Shakti_VERSION_MINOR 1)

project(Shakti)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

# List of build options.
option(SHAKTI_BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(SHAKTI_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(SHAKTI_BUILD_SAMPLES "Build samples in Shakti libraries" OFF)
option(SHAKTI_BUILD_TESTS "Build unit tests for Shakti libraries" OFF)


# Set default build type to Release mode.
set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type options are: Debug, Release")

set(BUILD_SHARED_LIBS ${SHAKTI_BUILD_SHARED_LIBS})


# ============================================================================ #
# State the list of dependencies.
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
find_package(DO_Sara COMPONENTS
  Core Graphics FeatureDescriptors ImageIO VideoIO REQUIRED)

# Use custom CMake macros.
set(DO_Shakti_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DO_Shakti_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/DO/Shakti)
include(shakti_macros)
include(shakti_configure_nvcc_compiler)
shakti_dissect_version()


# ============================================================================ #
# Build the following directories.
#
include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${DO_Sara_INCLUDE_DIRS}
  src
  third_party/gtest/include
)

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Turn off some annoying compilers for GCC
if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif ()

# Add some more compilation flags to build static libraries.
if (SHAKTI_USE_STATIC_LIBS OR NOT SHAKTI_BUILD_SHARED_LIBS)
  add_definitions("-DDO_SHAKTI_STATIC")
endif ()

add_subdirectory(third_party)
add_subdirectory(src)

if (SHAKTI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif ()

if (SHAKTI_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()


# ============================================================================ #
# Install the libraries.
include(shakti_installation_settings)
include(CPack)
