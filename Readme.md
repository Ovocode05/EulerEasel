# EulerEasel

It is a modular runtime for sparse matrix vector multiplication that adapts execution strategy based on hardware, uses matrix features as matrix context vector to predict the most suitable kernel for that matrix.
Rather than forcing the user to choose between formats and backends, the runtime automatically chooses the best memory layout, and hardware to perform spMV. spMV is a backbone of all the iterative algorithms like GRES, Conjugate Gradient etc. and it is repeated multiple times per iteration. Hence taking care of memory allocation and attaining the optimal performance and utilizing hardware has become the prime focus of many sparse-algebra libraries like Morpheus, Oracle, Ginkgo, etc.
## The problem

Sparse kernels are not one-size-fits-all.
Performance depends on:

- the structure of the matrix
- the available hardware
- the execution backend
- the size and shape of the workload
- the cost of configuration and setup

A static choice often leaves performance on the table.
The goal of EulerEasel is to reduce that brittleness by making execution decisions in a structured and adaptive way.

## The idea

EulerEasel is built around two distinct layers:

1. Capability layer
   - Determine which execution strategies are valid on the current machine.
   - Filter out impossible choices deterministically.
   - Example: remove CUDA-based strategies when GPU support is not available.

2. Decision layer
   - Choose the most suitable execution strategy for the current matrix and workload using one-step reinforcement learning.
   - Use hardware context, matrix features, and runtime state as input, and create nxn feature matrices for every kernel, where is the #features.
   - This is where online learning and policy selection are applied.
   - In one-step reinforcement learning, Linear Thompson sampling is implemented, due to its simplicity and accuracy level. After training over 50+ different matrices and similar matrices.
     <br>
     <br>
   ```
      {
        "uni_chimera_i5.mtx": {
            "best_kernel": "GPU_CSR",
            "runtime": 0.051023999229073524,
            "all_kernel_profiles": {
                "CPU_CSR": 0.6974400000000001,
                "CPU_ELL": 1.8573605,
                "CPU_CSR_AVX": 2.4386099999999997,
                "CPU_ELL_AVX_x4": 3.289846,
                "CPU_ELL_AVX_x16": 1.375461,
                "GPU_CSR": 0.051023999229073524,
                "GPU_ELL": 0.1085439994931221
            }
        }
    }
   ```
This is one example of matrix used for training with 100000 rows and columns with 299991 non zeros. Each kernel is run over 30 times for every matrix and a median of the performance list is calculated, selecting the best kernel based on runtime performance.

## Future Scope
- Use better Algorithms like Neural LinearTS, LinearUCB, or Deep RL etc.
- Using autotuner for optimizing the parameters of each kernels. eg. Threads, blocks or number of CPU cores (based on the availability). Methods include GridSearch, Bayesian Optimization, etc.
- Adding more backends like Triton, or making it adaptable to distributed GPUs, and hardware independent code using Kokkos. ( I mean there are thousands of other advance possibilities)
- Making use of hardware information like l1 cache misses, GFLOPS, or E-cores, P-cores. 

## Current repository scope

The repository currently includes:

- C++ sparse kernels and format-related utilities
- OpenMP and AVX-aware CPU paths
- a hardware context layer for CPU and GPU detection
- an optional CUDA path for GPU-enabled builds
- a CMake-based build system for local testing and development

## Build and run

### Requirements

- CMake 3.20+
- C++17 compiler
- OpenMP
- AVX2
- CUDA toolkit is optional, but required for the CUDA target

### Build with CUDA enabled

```bash
cd ~project_directory/root
rm -rf build
cmake -S . -B build -DEULER_BUILD_CUDA=ON
cmake --build build -j4
```

### Build without CUDA

```bash
cd ~project_directory/root
rm -rf build
cmake -S . -B build -DEULER_BUILD_CUDA=OFF
cmake --build build -j4
```

## Repository layout

- `Src/` — core implementation, kernels, runtime utilities, and strategy code and include pybind of cpp code
- `Data/` — sample matrices and synthetic inputs
- `Testing/` — test and validation scripts
- `build/` — generated build directory

<br>
[!Note]
I tried to make it like a python libraries using a pybind11 wrap over from-scratch written kernels in CUDA c++ ,using library openMP and AVX intrinsics for CPU based hardware. Imitating a small part of already existing techniques in large sparse algebra like Morpheus and Oracle.
