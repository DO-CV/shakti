find_package(NumPy 1.7.1 REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)

find_package(Boost COMPONENTS
             python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

file(GLOB shakti_SRC_FILES FILES *.cpp *.hpp)
python_add_module(shakti ${shakti_SRC_FILES})
target_link_libraries(shakti
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Shakti_ImageProcessing
)


set(VIRTUAL_ENV $ENV{VIRTUAL_ENV})

if (VIRTUAL_ENV)
  message(STATUS
    "Detected a python virtualenv in use, setting Python install prefix to "
    "to ${VIRTUAL_ENV}")

  set(Python_VERSION ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
  set(shakti_INSTALL_DIR
      ${VIRTUAL_ENV}/lib/python${Python_VERSION}/site-packages/do)

else ()
  if (PYTHON_VERSION_MAJOR STREQUAL "2")
    set(Python_VERSION ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
  else ()
    set(Python_VERSION ${PYTHON_VERSION_MAJOR})
  endif ()

  set(shakti_INSTALL_DIR
      /usr/lib/python${Python_VERSION}/dist-packages/do)

endif(VIRTUAL_ENV)

install(TARGETS shakti
        RUNTIME DESTINATION ${shakti_INSTALL_DIR}
        ARCHIVE DESTINATION ${shakti_INSTALL_DIR}
        LIBRARY DESTINATION ${shakti_INSTALL_DIR})
