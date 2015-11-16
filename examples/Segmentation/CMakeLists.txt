if (SARA_USE_VIDEOIO)
  find_package(DO_Sara COMPONENTS
    Core Graphics FeatureDescriptors ImageIO VideoIO REQUIRED)

  shakti_add_example(segmentation_example segmentation_example.cpp)
  target_link_libraries(
    segmentation_example
    ${DO_Sara_LIBRARIES}
    DO_Shakti_ImageProcessing
    DO_Shakti_Segmentation
    DO_Shakti_Utilities)
endif ()
