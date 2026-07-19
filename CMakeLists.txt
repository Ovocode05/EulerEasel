cmake_minimum_required(VERSION 3.20)
option(EULER_BUILD_CUDA "Build the CUDA demo target when CUDA is available" OFF)

if(EULER_BUILD_CUDA)
    find_program(GCC15_BIN gcc-15)
    find_program(GXX15_BIN g++-15)
    if(GCC15_BIN AND GXX15_BIN)
        message(STATUS "CUDA build requested: using gcc-15/g++-15 host compiler.")
        set(CMAKE_C_COMPILER "${GCC15_BIN}" CACHE FILEPATH "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "${GXX15_BIN}" CACHE FILEPATH "C++ compiler" FORCE)
        set(CMAKE_CUDA_HOST_COMPILER "${GCC15_BIN}" CACHE FILEPATH "CUDA host compiler" FORCE)
    else()
        message(STATUS "CUDA build requested but gcc-15/g++-15 not found; using default host compiler.")
    endif()
endif()

project(EulerEasel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(euler_common INTERFACE)

target_include_directories(euler_common INTERFACE
    ${CMAKE_SOURCE_DIR}/Src
    ${CMAKE_SOURCE_DIR}/Src/include
)

target_compile_features(euler_common INTERFACE cxx_std_17)

target_compile_options(euler_common INTERFACE
    -Wall
    -Wextra
    -mavx2
    -mfma
)

add_executable(euler_strategy
    Src/include/Strategy/strategy.cpp
)

target_link_libraries(euler_strategy PRIVATE euler_common OpenMP::OpenMP_CXX)

if(EULER_BUILD_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -Xcompiler=-fopenmp -Xcompiler=-mavx2 -Xcompiler=-mfma -Xcompiler=-march=native")
        enable_language(CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Xcompiler=-fopenmp -Xcompiler=-mavx2 -Xcompiler=-mfma -Xcompiler=-march=native")
        add_executable(euler_cuda
            Src/Server/CUDA/kernel.cu
        )

        target_include_directories(euler_cuda PRIVATE
            ${CMAKE_SOURCE_DIR}/Src
            ${CMAKE_SOURCE_DIR}/Src/include
            ${CMAKE_SOURCE_DIR}/Src/Server
        )

        target_compile_features(euler_cuda PRIVATE cxx_std_17)
        target_compile_definitions(euler_cuda PRIVATE HAS_CUDA=1)
        target_link_libraries(euler_cuda PRIVATE CUDA::cudart OpenMP::OpenMP_CXX stdc++)
        target_link_libraries(euler_cuda PRIVATE euler_common)
        set_target_properties(euler_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_ARCHITECTURES native
        )

        # Allow the CPU strategy executable to detect CUDA at runtime when CUDA is enabled.
        target_compile_definitions(euler_strategy PRIVATE HAS_CUDA=1)
        target_link_libraries(euler_strategy PRIVATE CUDA::cudart)
    else()
        message(STATUS "CUDA Toolkit not found; skipping CUDA demo target")
    endif()
endif()

add_custom_target(run_strategy
    COMMAND $<TARGET_FILE:euler_strategy>
    DEPENDS euler_strategy
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running the EulerEasel strategy demo"
)

if(TARGET euler_cuda)
    add_custom_target(run_cuda
        COMMAND $<TARGET_FILE:euler_cuda>
        DEPENDS euler_cuda
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running the EulerEasel CUDA demo"
    )
endif()

message(STATUS "EulerEasel configured. Primary build target: euler_strategy")
if(TARGET euler_cuda)
    message(STATUS "CUDA target available: euler_cuda")
else()
    message(STATUS "CUDA target disabled by default; run with -DEULER_BUILD_CUDA=ON if you want the CUDA demo")
endif()
