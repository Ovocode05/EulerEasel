#include <iostream>
#include <string>
using namespace std;

enum class Device
{
    CPU,
    GPU
};

enum class Architecture
{
    Proc,
    AVX,
    CUDA
};

enum class Format
{
    COO,
    CSR,
    ELL,
    HYB
};

enum class Kernel
{
    CPU_CSR,
    CPU_ELL,
    CPU_HYB,
    CPU_CSR_AVX,
    CPU_ELL_AVX_x4,
    CPU_ELL_AVX_x16,
    CPU_HYB_AVX,
    GPU_CSR,
    GPU_ELL,
    GPU_HYB
};

struct GPUParameters
{
    int32_t block_size = 0;
    int32_t threads_per_row = 0;
};

struct CPUParameters
{
    int32_t threads = 0;
    int32_t cores = 0;
};

struct HybridParameters
{
    int32_t ell_width = 0;
};

struct strategy
{
    Device device;
    Format format;
    Architecture arch;
    Kernel kernel;
    bool is_available = true;
};

struct ExecutionPlan
{
    Device device;
    Format format;
    Kernel kernel;

    GPUParameters gpu;
    CPUParameters cpu;
    HybridParameters hybrid;
};

struct ExecutionState
{
    Device matrix;
    Device input_vector;
    Device output_vector;
    Format current_format;
};