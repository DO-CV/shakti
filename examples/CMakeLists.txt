project(DO_Shakti_Examples)

message(STATUS "  - DO_Shakti_Examples")

file(GLOB cuda_test_SRC_FILES FILES *.cu *.cpp *.hpp)
shakti_add_example(cuda_test ${cuda_test_SRC_FILES})
target_link_libraries(
  cuda_test
  gcov
  ${DO_Sara_LIBRARIES}
  DO_Shakti_Utilities DO_Shakti_MultiArray)
