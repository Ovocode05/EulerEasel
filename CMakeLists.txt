cmake_minimum_required(VERSION 3.20)

option(EULER_BUILD_CUDA "Build with CUDA support" OFF)

# ============================================================
# CUDA toolchain discovery
#
# Must happen BEFORE enable_language(CUDA).
# ============================================================

if(EULER_BUILD_CUDA)

    # --------------------------------------------------------
    # 1. Find NVCC
    # --------------------------------------------------------

    find_program(
        EULER_NVCC
        NAMES nvcc
        HINTS
            /usr/local/cuda/bin
        PATHS
            /usr/local/cuda/bin
            /opt/cuda/bin
    )

    if(NOT EULER_NVCC)
        message(FATAL_ERROR
            "CUDA backend requested but nvcc was not found."
        )
    endif()


    # --------------------------------------------------------
    # 2. Find an NVCC-compatible host compiler
    # --------------------------------------------------------

    find_program(
        EULER_CUDA_HOST_COMPILER
        NAMES
            g++-15
            g++-14
            g++-13
    )

    if(NOT EULER_CUDA_HOST_COMPILER)
        message(FATAL_ERROR
            "CUDA backend requested but no compatible "
            "G++ host compiler was found."
        )
    endif()


    # --------------------------------------------------------
    # 3. Tell CMake exactly which compilers CUDA must use
    # --------------------------------------------------------

    set(
        CMAKE_CUDA_COMPILER
        "${EULER_NVCC}"
        CACHE FILEPATH
        "CUDA compiler"
        FORCE
    )

    set(
        CMAKE_CUDA_HOST_COMPILER
        "${EULER_CUDA_HOST_COMPILER}"
        CACHE FILEPATH
        "NVCC host compiler"
        FORCE
    )

endif()


# ============================================================
# Project
# ============================================================

project(EulerEasel LANGUAGES CXX)

if(EULER_BUILD_CUDA)
    enable_language(CUDA)

    message(STATUS "NVCC               : ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA host compiler : ${CMAKE_CUDA_HOST_COMPILER}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# ============================================================
# Common
# ============================================================

add_library(euler_common INTERFACE)

target_include_directories(euler_common INTERFACE
    ${CMAKE_SOURCE_DIR}/Src
    ${CMAKE_SOURCE_DIR}/Src/include
)

# Don't propagate CPU flags into NVCC compilation
target_compile_options(euler_common INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
    $<$<COMPILE_LANGUAGE:CXX>:-mfma>
)


add_library(euler_core INTERFACE)

target_link_libraries(euler_core INTERFACE
    euler_common
    OpenMP::OpenMP_CXX
)


# ============================================================
# Matrix extractor
# ============================================================

pybind11_add_module(matrix_extractor
    Src/include/bindings/bindings.cpp
)

target_link_libraries(matrix_extractor PRIVATE
    euler_core
)

target_include_directories(matrix_extractor PRIVATE
    ${CMAKE_SOURCE_DIR}/Src/Server
)

set_target_properties(matrix_extractor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Src/include
)


# ============================================================
# CPU runtime
# ============================================================

pybind11_add_module(runtime
    Src/include/bindings/runtimebind.cpp
)

target_link_libraries(runtime PRIVATE
    euler_core
)

target_include_directories(runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/Src/Server
)

set_target_properties(runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Src/include
)


# ============================================================
# CUDA runtime
# ============================================================

if(EULER_BUILD_CUDA)

    find_package(CUDAToolkit REQUIRED)
      # Hardware detection inside matrix_extractor uses CUDA Runtime API
    target_compile_definitions(matrix_extractor PRIVATE
        HAS_CUDA=1
    )

    target_link_libraries(matrix_extractor PRIVATE
        CUDA::cudart
    )

    pybind11_add_module(CUDAruntime
        Src/include/bindings/cudabindings.cpp
    )

    # cudabindings.cpp includes kernel.cuh, which contains
    # __global__, kernel launches <<<>>>, etc.
    # Therefore this translation unit MUST be compiled by NVCC.
    set_source_files_properties(
        Src/include/bindings/cudabindings.cpp
        PROPERTIES
        LANGUAGE CUDA
    )

    target_include_directories(CUDAruntime PRIVATE
        ${CMAKE_SOURCE_DIR}/Src
        ${CMAKE_SOURCE_DIR}/Src/include
        ${CMAKE_SOURCE_DIR}/Src/Server
        ${CMAKE_SOURCE_DIR}/Src/Server/CUDA
    )

    target_link_libraries(CUDAruntime PRIVATE
        CUDA::cudart
    )

    target_compile_definitions(CUDAruntime PRIVATE
        HAS_CUDA=1
    )

    set_target_properties(CUDAruntime PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES native

        LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/Src/include
    )

    add_custom_command(TARGET CUDAruntime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            make_directory "${CMAKE_SOURCE_DIR}/Python"

        COMMAND ${CMAKE_COMMAND} -E
            copy_if_different
            "$<TARGET_FILE:CUDAruntime>"
            "${CMAKE_SOURCE_DIR}/Python/$<TARGET_FILE_NAME:CUDAruntime>"
    )

endif()

foreach(module matrix_extractor runtime)
    add_custom_command(TARGET ${module} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            make_directory "${CMAKE_SOURCE_DIR}/Python"

        COMMAND ${CMAKE_COMMAND} -E
            copy_if_different
            "$<TARGET_FILE:${module}>"
            "${CMAKE_SOURCE_DIR}/Python/$<TARGET_FILE_NAME:${module}>"
    )
endforeach()

if(EULER_BUILD_CUDA)
    add_custom_command(TARGET CUDAruntime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            make_directory "${CMAKE_SOURCE_DIR}/Python"

        COMMAND ${CMAKE_COMMAND} -E
            copy_if_different
            "$<TARGET_FILE:CUDAruntime>"
            "${CMAKE_SOURCE_DIR}/Python/$<TARGET_FILE_NAME:CUDAruntime>"
    )
endif()


message(STATUS "")
message(STATUS "========== EulerEasel ==========")
message(STATUS "CPU runtime  : ON")
message(STATUS "CUDA runtime : ${EULER_BUILD_CUDA}")
message(STATUS "================================")