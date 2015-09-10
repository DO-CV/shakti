message(STATUS "")
do_step_message("Appending third-party libraries:")

set(THIRDPARTY_FOLDER_NAME "DO Shakti Third-Party")

# Disable any specific compiler flags for all third-party software.
set (CMAKE_C_FLAGS "")
set (CMAKE_CXX_FLAGS "")
if (UNIX AND DO_BUILD_SHARED_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif ()


if (SHAKTI_BUILD_TESTS)
  message(STATUS "  - Google Test")
  add_subdirectory(gtest)
  set_property(TARGET gtest PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})
  set_property(TARGET gtest_main PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})
endif ()
