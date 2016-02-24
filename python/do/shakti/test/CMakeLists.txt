# ============================================================================ #
# Find nosetests.
find_program(NOSETESTS_EXECUTABLE nosetests)
if(NOT NOSETESTS_EXECUTABLE)
  message(FATAL_ERROR "nosetests not found! Aborting...")
endif()


# ============================================================================ #
# Copy the Python bindings in the runtime output directory.
add_custom_target(copy_pyshakti_module)

# Create python module to the runtime output directory.
add_custom_command(TARGET copy_pyshakti_module
  COMMAND ${CMAKE_COMMAND} -E
  copy_directory ${DO_Shakti_DIR}/python/do
  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/do)

# Copy pyshakti.[so, dll] to the runtime output directory.
get_target_property(pyshakti_LIBRARY_PATH pyshakti LOCATION)
add_custom_command(TARGET copy_pyshakti_module
  COMMAND ${CMAKE_COMMAND} -E
  copy ${pyshakti_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})


# ============================================================================ #
# Run the unit tests.
add_custom_target(pytest
  COMMAND ${NOSETESTS_EXECUTABLE} --with-coverage --cover-package=do
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMENT "Running Python tests.")


# ============================================================================ #
# Make sure the python unit tests are copied first before running unit tests.
add_dependencies(pytest copy_pyshakti_module)
add_dependencies(copy_pyshakti_module pyshakti)
