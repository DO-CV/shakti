if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()


# Debug message.
shakti_step_message("FindDO_Shakti running for project '${PROJECT_NAME}'")


# 'find_package(DO_Shakti COMPONENTS Core Graphics ... REQUIRED)' is called.
set (DO_Shakti_ALL_COMPONENTS ImageProcessing MultiArray Utilities Segmentation)
if (DO_Shakti_FIND_COMPONENTS)
  set(DO_Shakti_USE_COMPONENTS ${SHAKTI_ALL_COMPONENTS})
else ()
  set(DO_Shakti_USE_COMPONENTS "")
  foreach (component ${DO_Shakti_FIND_COMPONENTS})
    list(FIND DO_Shakti_COMPONENTS ${component} COMPONENT_INDEX)
    if (COMPONENT_INDEX EQUAL -1)
      message (FATAL_ERROR "[Shakti] ${component} does not exist!")
    else ()
      list (APPEND DO_Shakti_USE_COMPONENTS ${component})
    endif ()
  endforeach (component)
endif ()


# Find include directories.
find_path(
  DO_Shakti_INCLUDE_DIRS
  NAMES DO/Shakti.hpp
  PATHS
  /usr/include /usr/local/include
  "C:/Program Files/DO-Shakti/include")


# Find compiled libraries.
foreach (COMPONENT ${DO_Shakti_USE_COMPONENTS})

  # Find compiled libraries.
  if (SHAKTI_USE_STATIC_LIBS)
    set (_library_name "DO_Shakti_${COMPONENT}-${DO_Shakti_VERSION}-s")
    set (_library_name_debug "DO_Shakti_${COMPONENT}-${DO_Shakti_VERSION}-sd")
  else ()
    set (_library_name "DO_Shakti_${COMPONENT}-${DO_Shakti_VERSION}")
    set (_library_name_debug "DO_Shakti_${COMPONENT}-${DO_Shakti_VERSION}-d")
  endif ()

  find_library(DO_Shakti_${COMPONENT}_DEBUG_LIBRARIES
    NAMES ${_library_name_debug}
    PATHS /usr/lib /usr/local/lib /opt/local/lib
          "C:/Program Files/DO-Shakti-Debug/lib"
    PATH_SUFFIXES DO/Shakti)

  find_library(DO_Shakti_${COMPONENT}_RELEASE_LIBRARIES
    NAMES ${_library_name}
    PATHS /usr/lib /usr/local/lib /opt/local/lib
          "C:/Program Files/DO-Shakti/lib"
    PATH_SUFFIXES DO/Shakti)

  if (NOT SHAKTI_USE_STATIC_LIBS AND NOT DO_Shakti_${COMPONENT}_DEBUG_LIBRARIES)
    set(
      DO_Shakti_${COMPONENT}_LIBRARIES
      ${DO_Shakti_${COMPONENT}_RELEASE_LIBRARIES}
      CACHE STRING "")
  else ()
    set(DO_Shakti_${COMPONENT}_LIBRARIES
      debug ${DO_Shakti_${COMPONENT}_DEBUG_LIBRARIES}
      optimized ${DO_Shakti_${COMPONENT}_RELEASE_LIBRARIES}
      CACHE STRING "")
  endif ()

  if (SHAKTI_USE_STATIC_LIBS)
    if (NOT DO_Shakti_${COMPONENT}_DEBUG_LIBRARIES OR
        NOT DO_Shakti_${COMPONENT}_RELEASE_LIBRARIES)
      message(FATAL_ERROR "DO_Shakti_${COMPONENT} is missing!")
    endif ()
  elseif (NOT DO_Shakti_${COMPONENT}_RELEASE_LIBRARIES)
    message(FATAL_ERROR "DO_Shakti_${COMPONENT} is missing!")
  endif ()

  if (DO_Shakti_${COMPONENT}_LIBRARIES)
    list(APPEND DO_Shakti_LIBRARIES ${DO_Shakti_${COMPONENT}_LIBRARIES})
  endif ()

endforeach()


# Load DO-specific macros
include(shakti_macros)
include(shakti_configure_nvcc_compiler)


# Specify DO-Shakti version.
include(DO_Shakti_version)


# List the compile flags needed by DO-Shakti.
if (SHAKTI_USE_STATIC_LIBS OR NOT SHAKTI_BUILD_SHARED_LIBS)
  add_definitions("-DDO_SHAKTI_STATIC")
endif ()


# Debug.
message("DO_Shakti_LIBRARIES = ${DO_Shakti_LIBRARIES}")
