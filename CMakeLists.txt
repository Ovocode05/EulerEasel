cmake_minimum_required(VERSION 3.20)

# Force standard C++14 language parsing for the NVCC compiler check
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 -allow-unsupported-compiler")

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

# Common Configurations
add_library(euler_common INTERFACE)
target_include_directories(euler_common INTERFACE
    ${CMAKE_SOURCE_DIR}/Src
    ${CMAKE_SOURCE_DIR}/Src/include
)
target_compile_options(euler_common INTERFACE
    -Wall -Wextra -O3 -march=native -mavx2 -mfma
)

# Core Target Framework
add_library(euler_core INTERFACE)
target_link_libraries(euler_core INTERFACE
    euler_common
    OpenMP::OpenMP_CXX
)

# ============================================================
# Matrix / strategy Python module
# ============================================================

pybind11_add_module(matrix_extractor
    Src/include/bindings/bindings.cpp
)

target_link_libraries(matrix_extractor PRIVATE euler_core)

target_include_directories(matrix_extractor PRIVATE
    ${CMAKE_SOURCE_DIR}/Src/Server
)

set_target_properties(matrix_extractor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Src/include
)


# ============================================================
# Runtime / SpMV Python module
# ============================================================

pybind11_add_module(runtime
    Src/include/bindings/runtimebind.cpp
)

target_link_libraries(runtime PRIVATE euler_core)

target_include_directories(runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/Src/Server
)

set_target_properties(runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Src/include
)

# Move production artifact out to Python folder cleanly
# Move Python modules to Python/ folder after build
add_custom_command(TARGET matrix_extractor POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/Python"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:matrix_extractor>"
        "${CMAKE_SOURCE_DIR}/Python/$<TARGET_FILE_NAME:matrix_extractor>"
)

add_custom_command(TARGET runtime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/Python"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:runtime>"
        "${CMAKE_SOURCE_DIR}/Python/$<TARGET_FILE_NAME:runtime>"
)
# Target-specific CUDA bindings configuration
if(EULER_BUILD_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(euler_common INTERFACE HAS_CUDA=1)
        target_link_libraries(euler_common INTERFACE CUDA::cudart)

        add_executable(euler_cuda Src/Server/CUDA/kernel.cu)
        target_include_directories(euler_cuda PRIVATE ${CMAKE_SOURCE_DIR}/Src/Server)
        target_link_libraries(euler_cuda PRIVATE euler_common OpenMP::OpenMP_CXX stdc++)
        
        set_target_properties(euler_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES native
        )
        message(STATUS "CUDA support enabled")
    else()
        message(STATUS "CUDA Toolkit not available; continuing without CUDA")
    endif()
endif()

message(STATUS "")
message(STATUS "========== EulerEasel ==========")
message(STATUS "Core library  : euler_core")
message(STATUS "Python target : euler (contains submodules)")
message(STATUS "CUDA build    : ${EULER_BUILD_CUDA}")
message(STATUS "================================")
