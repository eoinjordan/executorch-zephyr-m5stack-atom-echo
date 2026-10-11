# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
#
# "Hey Zephyr" wake word on the M5Stack ATOM Echo (ESP32-PICO-D4) under Zephyr.
# Same structure as the upstream zephyr/samples/hello-executorch sample: the
# model is embedded as a const array and only the operators it uses are linked.

cmake_minimum_required(VERSION 3.24)

set(CMAKE_SKIP_INSTALL_RULES ON CACHE BOOL "" FORCE)

set(ET_PTE_FILE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/model/kws.pte"
    CACHE FILEPATH "Path to the ExecuTorch .pte model to embed"
)
if(NOT IS_ABSOLUTE "${ET_PTE_FILE_PATH}")
  get_filename_component(
    ET_PTE_FILE_PATH "${ET_PTE_FILE_PATH}" ABSOLUTE BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(FATAL_ERROR "Model not found at ET_PTE_FILE_PATH=${ET_PTE_FILE_PATH}. "
                      "Run train/export_kws.py first or pass -DET_PTE_FILE_PATH=<file>.")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(atom_echo_hey_zephyr)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -ffunction-sections -fdata-sections"
)

if(NOT DEFINED EXECUTORCH_DIR)
  if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
    set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
  else()
    message(FATAL_ERROR "ExecuTorch Zephyr module not found. Check west.yml and run west update.")
  endif()
endif()
message(STATUS "EXECUTORCH_DIR=${EXECUTORCH_DIR}")

include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_DIR}/backends/arm/cmake/ArmRunnerUtils.cmake)

if(NOT TARGET portable_kernels)
  set(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY ON)
  add_subdirectory(
    ${EXECUTORCH_DIR}/kernels/portable
    ${CMAKE_CURRENT_BINARY_DIR}/executorch/kernels/portable
  )
  unset(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY)
endif()

arm_runner_create_selected_ops_lib(
  LIB_NAME "kws_portable_ops_lib"
  FUNCTIONS_YAML "${EXECUTORCH_DIR}/kernels/portable/functions.yaml"
  OPS_FROM_MODEL "${ET_PTE_FILE_PATH}"
  KERNEL_LIBS portable_kernels
  DEPS executorch
  DTYPE_SELECTIVE_BUILD ""
)

set(_local_flatcc_root ${CMAKE_BINARY_DIR}/flatcc_src)
if(NOT EXISTS ${_local_flatcc_root}/CMakeLists.txt)
  file(MAKE_DIRECTORY ${_local_flatcc_root})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${EXECUTORCH_DIR}/third-party/flatcc ${_local_flatcc_root}
  )
endif()
set(EXECUTORCH_FLATCC_SOURCE_ROOT ${_local_flatcc_root} CACHE PATH "" FORCE)
set(EXECUTORCH_FLATCC_INSTALL_ROOT ${_local_flatcc_root} CACHE PATH "" FORCE)

set(_model_pte_header ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h)
add_custom_command(
  OUTPUT ${_model_pte_header}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
          --pte ${ET_PTE_FILE_PATH} --out ${_model_pte_header}
  DEPENDS ${ET_PTE_FILE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pte_to_header.py
  COMMENT "Converting ${ET_PTE_FILE_PATH} to model_pte.h"
)
add_custom_target(gen_model_header DEPENDS ${_model_pte_header})
add_dependencies(app gen_model_header)

# The ExecuTorch module's runtime includes <zephyr/kernel.h>, which needs
# Zephyr's generated headers. Order it explicitly so a parallel build does not
# compile it before they exist.
if(TARGET zephyr_generated_headers)
  foreach(_t executorch_core executorch portable_kernels kernels_util_all_deps)
    if(TARGET ${_t})
      add_dependencies(${_t} zephyr_generated_headers)
    endif()
  endforeach()
endif()

target_sources(app PRIVATE src/main.cpp src/mic.c src/mfcc.c src/led.c)

target_compile_definitions(
  app PRIVATE
  APP_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE}
  APP_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE}
  MIC_RING_SAMPLES=16384
)

target_link_libraries(app PRIVATE libexecutorch kws_portable_ops_lib)
target_include_directories(app PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
