project(DO_Shakti_Examples)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


shakti_add_example(dense_sift_example dense_sift_example.cpp)
target_link_libraries(
  dense_sift_example
  ${DO_Sara_LIBRARIES}
  DO_Shakti_ImageProcessing
  DO_Shakti_Utilities)


if (SARA_USE_VIDEOIO)
  find_package(DO_Sara COMPONENTS
    Core Graphics FeatureDescriptors ImageIO VideoIO REQUIRED)

  shakti_add_example(dense_sift_example dense_sift_example.cpp)
  target_link_libraries(
    dense_sift_example
    ${DO_Sara_LIBRARIES}
    DO_Shakti_ImageProcessing
    DO_Shakti_Utilities)
endif ()
