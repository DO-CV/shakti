shakti_dissect_version(${DO_Shakti_VERSION})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(DO_Shakti_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DO/Shakti)


set(DO_Shakti_LIBRARIES MultiArray ImageProcessing Segmentation Utilities)
foreach (library ${DO_Shakti_LIBRARIES})
  shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/${library})
  shakti_create_common_variables("${library}")
  shakti_generate_library("${library}")
  target_link_libraries(DO_Shakti_${library} ${DO_Sara_LIBRARIES})
endforeach ()

target_link_libraries(DO_Shakti_ImageProcessing ${CUDA_CUDART_LIBRARY})
target_link_libraries(DO_Shakti_Utilities ${CUDA_CUDART_LIBRARY})
