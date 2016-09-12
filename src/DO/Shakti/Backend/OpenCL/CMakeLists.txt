list_subdirectories(DO_SHAKTI_BACKEND_OPENCL_FOLDERS ${CMAKE_CURRENT_SOURCE_DIR})

foreach (FOLDER ${DO_SHAKTI_BACKEND_OPENCL_FOLDERS})
  add_subdirectory(${FOLDER})
  set_property(TARGET ${FOLDER}
               PROPERTY FOLDER "Kojo Library")
endforeach ()