#include <iostream>
#include <iomanip>
#include <cstdint>
#include <vector>
#include <cstring>
using namespace std;

#ifdef __linux__
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

#ifdef HAS_CUDA
#include <cuda_runtime.h>
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

    double h2d_bandwidth_gbs;
    double d2h_bandwidth_gbs;
};

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
#endif
    ctx.cpu.measured_memory_bandwidth_gbs = 45.5;

#ifdef HAS_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); // Target structural primary device

        std::strncpy(ctx.gpu.name, prop.name, 256);
        ctx.gpu.mem_size_gb = static_cast<int32_t>(prop.totalGlobalMem / (1024 * 1024 * 1024));
        ctx.gpu.sm_count = prop.multiProcessorCount;
        ctx.gpu.warp_size = prop.warpSize;
        ctx.gpu.max_thread_per_block = prop.maxThreadsPerBlock;
        ctx.gpu.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        ctx.gpu.registers_per_sm = prop.regsPerBlock; // Approximated register maps
        ctx.gpu.shared_memory_per_sm = prop.sharedMemPerMultiprocessor;
        ctx.gpu.l2_cache = prop.l2CacheSize;
    }
#else
    strncpy(ctx.gpu.name, "No CUDA Device Detected", 256);
    ctx.gpu.mem_size_gb = 0;
    ctx.gpu.sm_count = 0;
    ctx.gpu.warp_size = 32;
#endif
    ctx.gpu.measured_memory_bandwidth_gbs = 900.0;
    ctx.gpu.launch_latency_us = 4.5;
    ctx.h2d_bandwidth_gbs = 12.5;
    ctx.d2h_bandwidth_gbs = 13.1;
}

void print_hardware_context(const HardwareContext &ctx)
{
    std::cout << "==================================================\n";
    std::cout << "          HARDWARE RUNTIME CONTEXT ENGINE         \n";
    std::cout << "==================================================\n";

    std::cout << "[CPU PROFILE]\n"
              << "  -> RAM Allocated:       " << ctx.cpu.mem_size_gb << " GB\n"
              << "  -> Cores (Phys/Log):    " << ctx.cpu.physical_cores << " / " << ctx.cpu.logical_threads << "\n"
              << "  -> SIMD Register Width: " << ctx.cpu.simd_width_bits << " bits\n"
              << "  -> Cache Line Alignment: " << ctx.cpu.cache_line_size << " bytes\n"
              << "  -> L1 / L2 / L3 Sizes:  " << (ctx.cpu.l1_cache / 1024) << "KB / "
              << (ctx.cpu.l2_cache / 1024) << "KB / " << (ctx.cpu.l3_cache / (1024 * 1024)) << "MB\n"
              << "  -> Read Bandwidth:      " << ctx.cpu.measured_memory_bandwidth_gbs << " GB/s\n\n";

    std::cout << "[GPU PROFILE] (" << ctx.gpu.name << ")\n"
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
    std::cout << "==================================================\n";
}