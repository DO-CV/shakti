message(STATUS "  - Building DO.Shakti examples")

add_subdirectory(ImageProcessing)
if (NOT WIN32)
  add_subdirectory(Quicktime)
endif ()
add_subdirectory(Segmentation)
