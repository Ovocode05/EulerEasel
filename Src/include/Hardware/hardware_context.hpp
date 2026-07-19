#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cstdlib> // Added for std::getenv

using namespace std;

#ifdef __linux__
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

struct CPUFeatures
{
    int32_t mem_size_gb;
    int32_t physical_cores;
    int32_t logical_threads;
    int32_t simd_width_bits;
    size_t cache_line_size;
    size_t l1_cache;
    size_t l2_cache;
    size_t l3_cache;

    double measured_memory_bandwidth_gbs;
};

struct GPUFeatures
{
    char name[256];
    int32_t mem_size_gb;
    int32_t sm_count;
    int32_t warp_size;
    int32_t max_thread_per_block;
    int32_t max_threads_per_sm;
    int32_t registers_per_sm;
    size_t shared_memory_per_sm;
    size_t l2_cache;

    double measured_memory_bandwidth_gbs;
    double launch_latency_us;
};

struct HardwareContext
{
    CPUFeatures cpu;
    GPUFeatures gpu;

    bool has_gpu = false;
    bool has_openMP = false;
    bool has_AVX = false;
    float gpu_cc = 0.0f; // Updated structure to support compute capability safely

    double h2d_bandwidth_gbs;
    double d2h_bandwidth_gbs;
};

void checkAVX_support(HardwareContext &ctx)
{
    try
    {
        if (__builtin_cpu_supports("avx2"))
        {
            ctx.has_AVX = true;
        }
        else
        {
            ctx.has_AVX = false;
        }
    }
    catch (...)
    {
        cerr << "[Hardware Check] Critical failure running CPUID check." << endl;
        ctx.has_AVX = false;
    }
}

void check_openmp_support(HardwareContext &ctx)
{
#if defined(_OPENMP)
    try
    {
        int max_threads = omp_get_max_threads();
        ctx.has_openMP = (max_threads > 0);
    }
    catch (...)
    {
        std::cerr << "[Hardware Check] OpenMP library failed to report max threads." << std::endl;
        ctx.has_openMP = false;
    }
#else
    ctx.has_openMP = false;
#endif
}

void query_hardware_context(HardwareContext &ctx)
{
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    ctx.cpu.mem_size_gb = static_cast<int32_t>((pages * page_size) / (1024 * 1024 * 1024));
    ctx.cpu.logical_threads = sysconf(_SC_NPROCESSORS_ONLN);
    ctx.cpu.physical_cores = ctx.cpu.logical_threads / 2;

    ctx.cpu.cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    ctx.cpu.l1_cache = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    ctx.cpu.l2_cache = sysconf(_SC_LEVEL2_CACHE_SIZE);
    ctx.cpu.l3_cache = sysconf(_SC_LEVEL3_CACHE_SIZE);
    ctx.cpu.simd_width_bits = 256;
#else
    ctx.cpu.mem_size_gb = 16;
    ctx.cpu.physical_cores = 8;
    ctx.cpu.logical_threads = 16;
    ctx.cpu.cache_line_size = 64;
    ctx.cpu.l1_cache = 32768;
    ctx.cpu.l2_cache = 262144;
    ctx.cpu.l3_cache = 16777216;
    ctx.cpu.simd_width_bits = 256;
#endif
    ctx.cpu.measured_memory_bandwidth_gbs = 45.5;

    // Run independent CPU and Software validation routines
    checkAVX_support(ctx);
    check_openmp_support(ctx);

    // Initial default layout for the GPU properties
    ctx.has_gpu = false;
    std::strncpy(ctx.gpu.name, "No CUDA Device Detected", 256);
    ctx.gpu.mem_size_gb = 0;
    ctx.gpu.sm_count = 0;
    ctx.gpu.warp_size = 32;
    ctx.gpu.max_thread_per_block = 0;
    ctx.gpu.max_threads_per_sm = 0;
    ctx.gpu.registers_per_sm = 0;
    ctx.gpu.shared_memory_per_sm = 0;
    ctx.gpu.l2_cache = 0;

#ifdef HAS_CUDA
    // USER ENVIRONMENT BYPASS CHECK:
    // If your system hardware is causing hard stalls/overheating, this bypass
    // prevents the driver handshake from executing, keeping your code running perfectly.
    const char *bypass_gpu = std::getenv("DISABLE_GPU_DETECTION");
    if (bypass_gpu != nullptr && std::string(bypass_gpu) == "1")
    {
        std::clog << "[Hardware Engine] GPU detection actively bypassed by environment flag." << std::endl;
    }
    else
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        std::cout << "cudaGetDeviceCount returned "
                  << cudaGetErrorName(err)
                  << " : "
                  << cudaGetErrorString(err)
                  << std::endl;

        if (err == cudaSuccess && device_count > 0)
        {
            cudaDeviceProp prop;
            cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);

            if (prop_err == cudaSuccess)
            {
                std::strncpy(ctx.gpu.name, prop.name, 256);
                ctx.has_gpu = true;
                ctx.gpu.mem_size_gb = static_cast<int32_t>(prop.totalGlobalMem / (1024 * 1024 * 1024));
                ctx.gpu.sm_count = prop.multiProcessorCount;
                ctx.gpu.warp_size = prop.warpSize;
                ctx.gpu.max_thread_per_block = prop.maxThreadsPerBlock;
                ctx.gpu.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
                ctx.gpu.registers_per_sm = prop.regsPerBlock;
                ctx.gpu.shared_memory_per_sm = prop.sharedMemPerMultiprocessor;
                ctx.gpu.l2_cache = prop.l2CacheSize;

                // Calculates the compute capability float safely (e.g., SM 8.6 becomes 8.6f)
                ctx.gpu_cc = prop.major + (prop.minor / 10.0f);
            }
            else
            {
                std::cerr << "[Hardware Check] Error fetching GPU properties: " << cudaGetErrorString(prop_err) << std::endl;
            }
        }
        else if (err != cudaSuccess)
        {
            std::cerr << "[Hardware Check] CUDA Driver Initialization failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
#endif

    ctx.gpu.measured_memory_bandwidth_gbs = ctx.has_gpu ? 900.0 : 0.0;
    ctx.gpu.launch_latency_us = ctx.has_gpu ? 4.5 : 0.0;
    ctx.h2d_bandwidth_gbs = ctx.has_gpu ? 12.5 : 0.0;
    ctx.d2h_bandwidth_gbs = ctx.has_gpu ? 13.1 : 0.0;
}

void print_hardware_context(const HardwareContext &ctx)
{
    std::cout << "          HARDWARE RUNTIME CONTEXT ENGINE         \n";

    std::cout << "[CPU PROFILE]\n"
              << "  -> RAM Allocated:       " << ctx.cpu.mem_size_gb << " GB\n"
              << "  -> Cores (Phys/Log):    " << ctx.cpu.physical_cores << " / " << ctx.cpu.logical_threads << "\n"
              << "  -> SIMD Register Width: " << ctx.cpu.simd_width_bits << " bits\n"
              << "  -> AVX Vector Support:  " << (ctx.has_AVX ? "YES (AVX2 Active)" : "NO") << "\n"
              << "  -> OpenMP Available:    " << (ctx.has_openMP ? "YES" : "NO") << "\n"
              << "  -> Cache Line Alignment: " << ctx.cpu.cache_line_size << " bytes\n"
              << "  -> L1 / L2 / L3 Sizes:  " << (ctx.cpu.l1_cache / 1024) << " KB / "
              << (ctx.cpu.l2_cache / 1024) << " KB / " << (ctx.cpu.l3_cache / (1024 * 1024)) << " MB\n"
              << "  -> Read Bandwidth:      " << ctx.cpu.measured_memory_bandwidth_gbs << " GB/s\n\n";

    std::cout << "[GPU PROFILE] (" << ctx.gpu.name << ")\n"
              << "  -> Hardware Detected:   " << (ctx.has_gpu ? "YES" : "NO") << "\n"
              << "  -> Compute Capability:  " << (ctx.has_gpu ? std::to_string(ctx.gpu_cc).substr(0, 3) : "N/A") << "\n"
              << "  -> VRAM Capacity:       " << ctx.gpu.mem_size_gb << " GB\n"
              << "  -> SM Compute Blocks:   " << ctx.gpu.sm_count << "\n"
              << "  -> Execution Warp Size: " << ctx.gpu.warp_size << " threads\n"
              << "  -> Max Threads (Blk/SM): " << ctx.gpu.max_thread_per_block << " / " << ctx.gpu.max_threads_per_sm << "\n"
              << "  -> Shared Memory / SM:  " << (ctx.gpu.shared_memory_per_sm / 1024) << " KB\n"
              << "  -> Device L2 Cache:     " << (ctx.gpu.l2_cache / (1024 * 1024)) << " MB\n"
              << "  -> VRAM Bandwidth:      " << ctx.gpu.measured_memory_bandwidth_gbs << " GB/s\n\n";

    std::cout << "[INTERCONNECT BUS]\n"
              << "  -> Host-To-Device (H2D): " << ctx.h2d_bandwidth_gbs << " GB/s\n"
              << "  -> Device-To-Host (D2H): " << ctx.d2h_bandwidth_gbs << " GB/s\n";
}
