shakti_dissect_version(${DO_Shakti_VERSION})


# Set output directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Create the following libraries.
set(DO_Shakti_LIBRARIES MultiArray ImageProcessing Segmentation Utilities)
foreach (library ${DO_Shakti_LIBRARIES})
  shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/${library})
  shakti_create_common_variables("${library}")
  shakti_generate_library("${library}")
  target_link_libraries(DO_Shakti_${library}
    ${DO_Sara_Core_LIBRARIES}
    ${CUDA_CUDART_LIBRARY})
endforeach ()


# Python bindings.
if (SHAKTI_BUILD_PYTHON_BINDINGS)
  add_subdirectory(DO/Shakti/Python)
endif ()


# Install licenses and source files.
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  install(FILES ${DO_Shakti_DIR}/COPYING.README
                ${DO_Shakti_DIR}/COPYING.MPL2
    DESTINATION include/DO/Shakti
    COMPONENT Sources)
  install(DIRECTORY ${DO_Shakti_DIR}/cmake
    DESTINATION include/DO/Shakti
    COMPONENT Sources)
  install(DIRECTORY ${DO_Shakti_DIR}/src/DO
    DESTINATION include/
    COMPONENT Sources)
  install(FILES ${DO_Shakti_SOURCE_DIR}/Defines.hpp
    DESTINATION include/DO/Shakti
    COMPONENT Sources)
  set(CPACK_COMPONENT_Sources_REQUIRED 1)
endif ()
