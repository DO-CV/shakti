if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()


# Load DO-specific macros
include(shakti_macros)


# Specify DO-Shakti version.
include(DO_Shakti_version)


# Debug message.
shakti_step_message("FindDO_Shakti running for project '${PROJECT_NAME}'")


# Setup DO-CV once for all for every test projects in the 'test' directory.
if (NOT DO_Shakti_FOUND)

  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDO_Shakti.cmake")
    message(STATUS "Building DO-Shakti from source")

    # Convenience variables used later in 'UseDOShaktiXXX.cmake' scripts.
    set(DO_Shakti_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "")
    set(DO_Shakti_INCLUDE_DIR ${DO_Shakti_DIR}/src CACHE STRING "")
    set(DO_Shakti_SOURCE_DIR ${DO_Shakti_DIR}/src/DO/Shakti)
    set(DO_Shakti_ThirdParty_DIR ${DO_Shakti_DIR}/third-party CACHE STRING "")

    message("DO_Shakti_SOURCE_DIR = ${DO_Shakti_SOURCE_DIR}")

  endif ()

endif ()
