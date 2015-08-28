project(DO_Shakti_Examples)

message(STATUS "  - DO_Shakti_Examples")

shakti_add_example(segmentation_example segmentation_example.cpp)
target_link_libraries(
  segmentation_example
  ${DO_Sara_LIBRARIES}
  DO_Shakti_ImageProcessing
  DO_Shakti_Segmentation
  DO_Shakti_Utilities
)
if (UNIX)
  target_link_libraries(segmentation_example gcov)
endif ()
