cmake_minimum_required(VERSION 2.8.12)

project(Shakti)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()


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
include(convenience_macros)


# Set compiler flags and include directories.
if (NOT WIN32)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
endif ()

include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${DO_Sara_INCLUDE_DIRS}
  src
)


# Build the following directories.
add_subdirectory(examples)
add_subdirectory(src)
#add_subdirectory(test)
#add_subdirectory(third-party)
