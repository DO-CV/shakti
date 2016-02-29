# ============================================================================ #
# Find the needed third-party libraries.
find_package(NumPy 1.7.1 REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)

if (PYTHON_VERSION_MAJOR STREQUAL 3)
  set(SHAKTI_BOOST_PYTHON python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
else ()
  set(SHAKTI_BOOST_PYTHON python)
endif ()
find_package(Boost COMPONENTS ${SHAKTI_BOOST_PYTHON} REQUIRED)


# ============================================================================ #
# Compile the library.
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

file(GLOB pyshakti_SRC_FILES FILES *.cpp)

python_add_module(pyshakti ${pyshakti_SRC_FILES})

target_link_libraries(pyshakti
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Shakti_ImageProcessing)


# ============================================================================ #
# Install the library in the currently active python environment.
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "from distutils.sysconfig import get_python_lib; print get_python_lib()"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message("Installing shakti Python bindings to ${PYTHON_SITE_PACKAGES_DIR}")

install(TARGETS pyshakti
        RUNTIME DESTINATION ${PYTHON_SITE_PACKAGES_DIR}
        ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES_DIR}
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES_DIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../shakti
        DESTINATION ${PYTHON_SITE_PACKAGES_DIR}/do)


if (SHAKTI_BUILD_TESTS AND SHAKTI_BUILD_PYTHON_BINDINGS)
  add_subdirectory(test)
endif ()
