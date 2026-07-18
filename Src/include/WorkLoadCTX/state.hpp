#include <iostream>
#include <string>
using namespace std;

enum class Device
{
    CPU,
    GPU
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
    Kernel kernel;
    bool requires_conversion;
    bool requires_transfer;
    bool feasiblel;
    string reject_reason;
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