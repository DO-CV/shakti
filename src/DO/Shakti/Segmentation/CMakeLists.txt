file(GLOB Segmentation_SRC_FILES FILES *.hpp *.cu)

cuda_add_library(DO_Shakti_Segmentation
  ../Segmentation.hpp
  ${Segmentation_SRC_FILES})
set_property(TARGET DO_Shakti_Segmentation PROPERTY FOLDER "DO Shakti Libraries")
