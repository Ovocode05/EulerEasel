# EulerEasel

Sparse linear algebra is everywhere, but its performance is rarely automatic.
A matrix that runs well on one machine may behave very differently on another.
The same kernel may be ideal for one structure and poor for another.
In practice, many systems still rely on fixed rules, hand-picked formats, and manual tuning.

EulerEasel is a different approach.
It is a modular runtime for sparse linear algebra that adapts execution strategy based on hardware, matrix structure, and workload context.
Rather than forcing the user to choose between formats, backends, and parameter settings manually, the runtime reasons about what is possible, what is likely to work well, and how to configure execution for the current situation.

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

EulerEasel is built around three distinct layers:

1. Capability layer
   - Determine which execution strategies are valid on the current machine.
   - Filter out impossible choices deterministically.
   - Example: remove CUDA-based strategies when GPU support is not available.

2. Decision layer
   - Choose the most suitable execution strategy for the current matrix and workload.
   - Use hardware context, matrix features, and runtime state as input.
   - This is where online learning and policy selection are applied.

3. Configuration layer
   - Tune kernel-specific parameters such as block size, thread count, and layout choices.
   - Optimize the selected strategy rather than treating all configurations as equivalent.

This separation is important.
It makes the system easier to reason about: first determine what is possible, then decide what should be used, then refine how it should be run.

## Why this is interesting

The novelty of EulerEasel is not simply that it uses GPUs, AVX, OpenMP, or CUDA.
Those are implementation tools.
The real contribution is the runtime perspective:

- it treats execution as a decision problem rather than a fixed pipeline
- it combines deterministic filtering with adaptive policy selection
- it aims to reduce manual tuning and static heuristics
- it is designed to grow from a single runtime strategy into a broader adaptive sparse execution system

In short, EulerEasel is an attempt to make sparse linear algebra execution more automatic, more context-aware, and more robust across changing conditions.

## Vision

The long-term vision is a runtime that can:

- analyze sparse workloads at runtime
- select an appropriate execution strategy for each context
- adapt as hardware and workloads change
- learn from observed performance over time
- provide a cleaner path from raw sparse computation to efficient execution without requiring the user to manually engineer every choice

The project is meant to evolve from a small experimental runtime into a general adaptive execution framework for sparse linear algebra.

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
- CUDA toolkit is optional, but required for the CUDA target

### Build with CUDA enabled

```bash
cd /home/fakeheadset/Projects/EulerEasel
rm -rf build
cmake -S . -B build -DEULER_BUILD_CUDA=ON
cmake --build build -j4
```

### Build without CUDA

```bash
cd /home/fakeheadset/Projects/EulerEasel
rm -rf build
cmake -S . -B build -DEULER_BUILD_CUDA=OFF
cmake --build build -j4
```

### Run the executables

```bash
./build/euler_strategy
./build/euler_cuda
```

## Repository layout

- `Src/` — core implementation, kernels, runtime utilities, and strategy code
- `Data/` — sample matrices and synthetic inputs
- `Testing/` — test and validation scripts
- `build/` — generated build directory

## Development direction

The project is intended to evolve in a practical way:

- first, make capability detection reliable
- then improve runtime decision making
- then add richer parameter tuning and adaptation

That keeps the scope focused while still leaving room for a serious systems project to grow.
