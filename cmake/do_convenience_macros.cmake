function (shakti_add_example)
   # Get the test executable name.
   list(GET ARGN 0 EXAMPLE_NAME)
   message(STATUS "EXAMPLE NAME = ${EXAMPLE_NAME}")

   # Get the list of source files.
   list(REMOVE_ITEM ARGN ${EXAMPLE_NAME})
   message(STATUS "SOURCE FILES = ${ARGN}")

   # Split the list of source files in two sub-lists:
   # - list of CUDA source files.
   # - list of regular C++ source files.
   set (CUDA_SOURCE_FILES "")
   set (CPP_SOURCE_FILES "")
   foreach (SOURCE ${ARGN})
     if (${SOURCE} MATCHES "(.*).cu$")
       list(APPEND CUDA_SOURCE_FILES ${SOURCE})
     else ()
       list(APPEND CPP_SOURCE_FILES ${SOURCE})
     endif()
   endforeach ()
   message(STATUS "CUDA_SOURCE_FILES = ${CUDA_SOURCE_FILES}")
   message(STATUS "CPP_SOURCE_FILES = ${CPP_SOURCE_FILES}")

   # Add the C++ test executable.
   add_executable(${EXAMPLE_NAME} ${CPP_SOURCE_FILES})
   set_property(TARGET ${EXAMPLE_NAME} PROPERTY FOLDER "DO Shakti Examples")
   set_target_properties(
     ${EXAMPLE_NAME} PROPERTIES
     COMPILE_FLAGS ${DO_DEFINITIONS}
   )

   # Create an auxilliary library for CUDA based code.
   # This is a workaround to do unit-test CUDA code with gtest.
   if (NOT "${CUDA_SOURCE_FILES}" STREQUAL "")
     source_group("CUDA Source Files" REGULAR_EXPRESSION ".*\\.cu$")
     cuda_add_library(${EXAMPLE_NAME}_CUDA_AUX ${CUDA_SOURCE_FILES})
     target_link_libraries(${EXAMPLE_NAME}_CUDA_AUX
                           DO_Shakti_Utilities
                           ${DO_LIBRARIES})
     # Group the unit test in the "Tests" folder.
     set_property(
       TARGET ${EXAMPLE_NAME}_CUDA_AUX PROPERTY FOLDER "CUDA Examples")

     target_link_libraries(${EXAMPLE_NAME} ${EXAMPLE_NAME}_CUDA_AUX ${DO_LIBRARIES})
   endif ()
endfunction ()


