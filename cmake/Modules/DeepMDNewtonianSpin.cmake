# SPDX-License-Identifier: LGPL-3.0-or-later

set(DEEPMD_SOURCE_ROOT "" CACHE PATH "DeePMD-kit source tree for the built-in Newtonian spin adapter")
if(NOT DEEPMD_SOURCE_ROOT)
  return()
endif()

if(NOT PKG_SPIN OR NOT PKG_KOKKOS)
  message(FATAL_ERROR "DEEPMD_SOURCE_ROOT requires PKG_SPIN=ON and PKG_KOKKOS=ON")
endif()

find_package(Git REQUIRED)
get_filename_component(_deepmd_source_root "${DEEPMD_SOURCE_ROOT}" REALPATH)
set(_deepmd_lmp_dir "${_deepmd_source_root}/source/lmp")
foreach(_required_file builtin.cmake deepmd_version.h.in pair_dpa4spin_kokkos.cpp pair_dpa4spin_kokkos.h)
  if(NOT EXISTS "${_deepmd_lmp_dir}/${_required_file}")
    message(FATAL_ERROR "DEEPMD_SOURCE_ROOT is missing source/lmp/${_required_file}: ${_deepmd_source_root}")
  endif()
endforeach()

set(DEEPMD_ADAPTER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/deepmd-newtonian-spin")
cmake_path(IS_PREFIX DEEPMD_ADAPTER_BINARY_DIR "${_deepmd_lmp_dir}" NORMALIZE _source_in_adapter_dir)
if(_source_in_adapter_dir)
  message(FATAL_ERROR "DEEPMD_SOURCE_ROOT must be outside the generated adapter directory: ${DEEPMD_ADAPTER_BINARY_DIR}")
endif()

set(_deepmd_patch "${CMAKE_CURRENT_LIST_DIR}/../patches/deepmd-newtonian-spin.patch")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_deepmd_patch}")
file(GLOB _deepmd_adapter_files CONFIGURE_DEPENDS "${_deepmd_lmp_dir}/*.cpp" "${_deepmd_lmp_dir}/*.h")
list(APPEND _deepmd_adapter_files "${_deepmd_lmp_dir}/builtin.cmake" "${_deepmd_lmp_dir}/deepmd_version.h.in")

# Recreate the generated copy so reconfiguration cannot retain stale sources
# or apply the patch twice. The external DeePMD-kit tree is read-only input.
file(REMOVE_RECURSE "${DEEPMD_ADAPTER_BINARY_DIR}")
file(MAKE_DIRECTORY "${DEEPMD_ADAPTER_BINARY_DIR}")
foreach(_source_file IN LISTS _deepmd_adapter_files)
  get_filename_component(_filename "${_source_file}" NAME)
  configure_file("${_source_file}" "${DEEPMD_ADAPTER_BINARY_DIR}/${_filename}" COPYONLY)
endforeach()

# Stop Git discovery before it reaches the LAMMPS repository when the build
# directory is inside the checkout. Only the generated copy is patched.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE --unset=GIT_INDEX_FILE
          "GIT_CEILING_DIRECTORIES=${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles"
          "${GIT_EXECUTABLE}" apply --whitespace=error "${_deepmd_patch}"
  WORKING_DIRECTORY "${DEEPMD_ADAPTER_BINARY_DIR}"
  RESULT_VARIABLE _patch_result
  OUTPUT_VARIABLE _patch_output
  ERROR_VARIABLE _patch_error)
if(NOT _patch_result EQUAL 0)
  message(FATAL_ERROR
    "The Newtonian spin patch does not apply to ${_deepmd_lmp_dir}. "
    "Use a compatible DeePMD-kit source revision matching the installed libraries. "
    "The external source tree has not been modified.\n${_patch_output}${_patch_error}")
endif()

message(STATUS "Built-in Newtonian spin adapter: ${DEEPMD_ADAPTER_BINARY_DIR}")
include("${DEEPMD_ADAPTER_BINARY_DIR}/builtin.cmake")
