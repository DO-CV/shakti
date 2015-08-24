file(GLOB_RECURSE ImageProcessing_SRC_FILES FILES *.hpp *.cu)

cuda_add_library(DO_Shakti_ImageProcessing ${ImageProcessing_SRC_FILES})
set_property(
  TARGET DO_Shakti_ImageProcessing
  PROPERTY
  FOLDER "DO Shakti Libraries")
