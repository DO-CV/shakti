project(DO_Shakti_Examples)

message(STATUS "  - DO_Shakti_Examples")

file(GLOB image_processing_SRC_FILES FILES *.cu *.cpp  *.hpp)
shakti_add_example(image_processing_test ${image_processing_SRC_FILES})
target_link_libraries(
  image_processing_test
  ${DO_Sara_LIBRARIES}
  DO_Shakti_Utilities
  DO_Shakti_MultiArray
)
if (UNIX)
  target_link_libraries(image_processing_test gcov)
endif ()
