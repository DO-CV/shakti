# Turn off some annoying compilers for GCC.
if (CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif ()

# Add some more compilation flags to build static libraries.
if (SHAKTI_USE_STATIC_LIBS OR NOT SHAKTI_BUILD_SHARED_LIBS)
  add_definitions("-DDO_SHAKTI_STATIC")
endif ()
