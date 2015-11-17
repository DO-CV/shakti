find_package(NumPy 1.7.1 REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)

if (SHAKTI_BUILD_SPECIFIC_PYTHON)
  set(SHAKTI_BOOST_PYTHON python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
else ()
  set(SHAKTI_BOOST_PYTHON python)
endif ()
find_package(Boost COMPONENTS ${SHAKTI_BOOST_PYTHON} REQUIRED)


include_directories(
  ${Boost_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

file(GLOB shakti_SRC_FILES FILES *.cpp *.hpp)
python_add_module(shakti ${shakti_SRC_FILES})
target_link_libraries(shakti
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Shakti_ImageProcessing
)


execute_process(COMMAND python -c
  "from distutils.sysconfig import get_python_lib; print get_python_lib()"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(shakti_INSTALL_DIR ${PYTHON_SITE_PACKAGES_DIR}/do)

message("Installing Shakti Python bindings to ${shakti_INSTALL_DIR}")

install(TARGETS shakti
        RUNTIME DESTINATION ${shakti_INSTALL_DIR}
        ARCHIVE DESTINATION ${shakti_INSTALL_DIR}
        LIBRARY DESTINATION ${shakti_INSTALL_DIR})
