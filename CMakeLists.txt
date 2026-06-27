cmake_minimum_required(VERSION 3.12)
project(eulereasel)

set(CMAKE_CXX_STANDARD 17)
cmake_policy(SET CMP0148 NEW)

# 1. Python environment path
set(pybind11_DIR "/home/fakeheadset/envs/ai/lib64/python3.14/site-packages/pybind11/share/cmake/pybind11")

# 2. Find dependencies
find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

# 3. Include directories
include_directories(
    Src/Server/CPU
    Src/Server/CPU/utils
)

# 4. Define your module name (Using 'eulereasel' everywhere)
pybind11_add_module(eulereasel
    Src/Server/CPU/binding.cpp
)

# 5. Link libraries and compile options to 'eulereasel'
target_link_libraries(eulereasel PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(eulereasel PRIVATE /arch:AVX2 /O2)
else()
    target_compile_options(eulereasel PRIVATE -mavx2 -O3 -march=native)
endif()
