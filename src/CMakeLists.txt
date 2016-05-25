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


if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  configure_file(${DO_Shakti_DIR}/cmake/DO_ShaktiConfig.cmake.in
                 ${CMAKE_BINARY_DIR}/cmake/DO_ShaktiConfig.cmake @ONLY)
  file(COPY ${DO_Shati_DIR}/cmake DESTINATION ${CMAKE_BINARY_DIR})

  # Install license files.
  install(FILES ${DO_Shakti_DIR}/COPYING.README
                ${DO_Shakti_DIR}/COPYING.MPL2
          DESTINATION ${SHAKTI_INSTALL_DIR}/include/DO/Shakti
          COMPONENT Sources)

  # Install CMake scripts.
  install(DIRECTORY ${CMAKE_BINARY_DIR}/cmake
          DESTINATION ${SHAKTI_INSTALL_DIR}/share/DO_Shakti
          COMPONENT Sources)

  # Install source files.
  install(FILES ${DO_Shakti_SOURCE_DIR}/Defines.hpp
          DESTINATION ${SHAKTI_INSTALL_DIR}/include/DO/Shakti
          COMPONENT Sources)

  install(DIRECTORY ${DO_Shakti_DIR}/src/DO
          DESTINATION ${SHAKTI_INSTALL_DIR}/include
          COMPONENT Sources)

  set(CPACK_COMPONENT_Sources_REQUIRED 1)
endif ()
