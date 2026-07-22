cmake_minimum_required(VERSION 3.20)

option(EULER_BUILD_CUDA "Build with CUDA support" OFF)

if(EULER_BUILD_CUDA)
    find_program(NVCC_BIN NAMES nvcc PATHS /usr/local/cuda/bin /usr/local/cuda-*/bin ENV PATH)

    if(NOT NVCC_BIN)
        message(WARNING "EULER_BUILD_CUDA=ON but no nvcc compiler was found; disabling CUDA support")
        set(EULER_BUILD_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)
    elseif(NOT CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "${NVCC_BIN}" CACHE FILEPATH "Path to nvcc" FORCE)
    endif()
endif()

project(EulerEasel LANGUAGES CXX)

if(EULER_BUILD_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

############################################################
# Common interface
############################################################

add_library(euler_common INTERFACE)

target_include_directories(euler_common INTERFACE
    ${CMAKE_SOURCE_DIR}/Src
    ${CMAKE_SOURCE_DIR}/Src/include
)

target_compile_options(euler_common INTERFACE
    -Wall
    -Wextra
    -O3
    -march=native
    -mavx2
    -mfma
)

############################################################
# Core runtime library
############################################################

add_library(euler_core INTERFACE)

target_link_libraries(euler_core
INTERFACE
    euler_common
    OpenMP::OpenMP_CXX
)

############################################################
# Python modules
############################################################

set(EULER_PYTHON_MODULES matrix_extractor filter)

foreach(module_name IN LISTS EULER_PYTHON_MODULES)
    pybind11_add_module(${module_name}
        Src/include/bindings.cpp
    )

    target_link_libraries(${module_name}
    PRIVATE
        euler_core
    )

    target_include_directories(${module_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/Src
        ${CMAKE_SOURCE_DIR}/Src/include
        ${CMAKE_SOURCE_DIR}/Src/Server
    )

    target_compile_definitions(${module_name} PRIVATE
        EULER_PYBIND_MODULE_NAME=${module_name}
    )

    set_target_properties(${module_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Src/include
    )

    add_custom_command(TARGET ${module_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/Python"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${module_name}>"
            "${CMAKE_SOURCE_DIR}/Python/$<TARGET_FILE_NAME:${module_name}>"
    )
endforeach()

############################################################
# CUDA
############################################################

if(EULER_BUILD_CUDA)

    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)

        enable_language(CUDA)

        target_compile_definitions(euler_common
        INTERFACE
            HAS_CUDA=1
        )

        target_link_libraries(euler_common
        INTERFACE
            CUDA::cudart
        )

        add_executable(euler_cuda
            Src/Server/CUDA/kernel.cu
        )

        target_include_directories(euler_cuda PRIVATE
            ${CMAKE_SOURCE_DIR}/Src
            ${CMAKE_SOURCE_DIR}/Src/include
            ${CMAKE_SOURCE_DIR}/Src/Server
        )

        target_link_libraries(euler_cuda PRIVATE
            euler_common
            CUDA::cudart
            OpenMP::OpenMP_CXX
            stdc++
        )

        set_target_properties(euler_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES native
        )

        message(STATUS "CUDA support enabled")

    else()

        message(STATUS "CUDA Toolkit or CUDA compiler not available; continuing without CUDA")

    endif()

endif()

############################################################
# Status
############################################################

message(STATUS "")
message(STATUS "========== EulerEasel ==========")
message(STATUS "Python module : matrix_extractor")
message(STATUS "Core library  : euler_core")

if(EULER_BUILD_CUDA)
    message(STATUS "CUDA build    : ON")
else()
    message(STATUS "CUDA build    : OFF")
endif()

message(STATUS "================================")