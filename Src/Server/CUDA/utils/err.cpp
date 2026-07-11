#include <iostream>
#include <cudaruntime.h>
#include <cuda/cmath>
using namespace std;

inline void cudaCheck(
    cudaError_t err,
    const char *file,
    int line, )
{
    if (err != cudaSuccess)
    {
        throw runtime_error(
            string(Cuda Err at) +
            file + ":" +
            to_string(line) + ":" +
            cudaGetErrorString(err));
    }
}

#define CUDA_CHECK(call) \
    cudaCheck((call), __FILE__, __LINE__)