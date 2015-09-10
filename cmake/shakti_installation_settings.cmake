# List all available components for installation.
set(CPACK_COMPONENTS_ALL Sources Libraries)


if (WIN32)
  set(CPACK_PACKAGE_NAME "DO-Shakti")
else()
  set(CPACK_PACKAGE_NAME "libDO-Shakti")
endif ()
if (DO_BUILD_SHARED_LIBS)
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-shared")
else ()
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-static")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-dbg")
endif ()

set(CPACK_PACKAGE_VENDOR "DO-CV")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "DO-Shakti: C++11/CUDA-accelerated Computer Vision")
set(CPACK_RESOURCE_FILE_LICENSE "${DO_Shakti_DIR}/COPYING.README")
set(CPACK_PACKAGE_CONTACT "David OK")

set(CPACK_PACKAGE_VERSION_MAJOR ${DO_Shakti_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${DO_Shakti_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${DO_Shakti_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${DO_Shakti_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO-Shakti")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO-Shakti-${CMAKE_BUILD_TYPE}")
endif ()



# ============================================================================ #
# Special configuration for Debian packages.
#
set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "cmake")



# ============================================================================ #
# Special configuration for Windows installer using NSIS.
#
# Installers for 32- vs. 64-bit CMake:
#  - Root install directory (displayed to end user at installer-run time)
#  - "NSIS package/display name" (text used in the installer GUI)
#  - Registry key used to store info about the installation
if(CMAKE_CL_64)
  set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  set(CPACK_NSIS_PACKAGE_NAME
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win64")
  set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win64")
else()
  set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
  set(CPACK_NSIS_PACKAGE_NAME
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win32")
  set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win32")
endif()
set(CPACK_NSIS_COMPRESSOR "/SOLID lzma")

set(CPACK_NSIS_DISPLAY_NAME ${CPACK_NSIS_PACKAGE_NAME})

# ============================================================================ #
# Select package generator.
if (WIN32)
  set(CPACK_GENERATOR NSIS)
elseif (UNIX)
  set(CPACK_GENERATOR "DEB")
endif()
