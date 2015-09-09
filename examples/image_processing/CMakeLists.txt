project(DO_Shakti_Examples)

message(STATUS "  - DO_Shakti_Examples")

file(GLOB image_processing_SRC_FILES FILES *.cpp  *.hpp)
shakti_add_example(image_processing_example ${image_processing_SRC_FILES})
target_link_libraries(
  image_processing_example
  ${DO_Sara_LIBRARIES}
  DO_Shakti_ImageProcessing
  DO_Shakti_MultiArray
  DO_Shakti_Utilities
)
if (UNIX)
  target_link_libraries(image_processing_example gcov)
endif ()
